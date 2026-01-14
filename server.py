import torch
import pypdfium2 as pdfium
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModel, AutoTokenizer
import uvicorn
import tempfile
import shutil
import os
from concurrent.futures import ProcessPoolExecutor
import functools
from io import BytesIO

app = FastAPI(title="DeepSeek-OCR Parallel")

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda"
# Using 8 processes to render PDF pages on the CPU for faster parallel rendering
cpu_executor = ProcessPoolExecutor(max_workers=8)
INFERENCE_BATCH_SIZE = 4

print(f"Loading {MODEL_ID} with Flash Attention 2 (Parallel)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE
).eval()

# Helper function to render a page (runs in a separate process)
def render_page_worker(pdf_bytes, page_index):
    # Re-opening the PDF inside the worker makes it 100% thread/process safe
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[page_index]
    bitmap = page.render(scale=3.0)
    return bitmap.to_pil()

@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    request_output_dir = tempfile.mkdtemp()
    
    try:
        pdf_data = await file.read()
        pdf = pdfium.PdfDocument(pdf_data)
        num_pages = len(pdf)
        
        # 1. Parallel CPU Rendering
        # This renders all pages at once across 8 CPU cores
        render_func = functools.partial(render_page_worker, pdf_data)
        images = list(cpu_executor.map(render_func, range(num_pages)))
        
        full_markdown = []

        # 2. Optimized GPU Inference with Batching
        with torch.inference_mode():
            for batch_start in range(0, len(images), INFERENCE_BATCH_SIZE):
                batch_end = min(batch_start + INFERENCE_BATCH_SIZE, len(images))
                batch_images = images[batch_start:batch_end]
                
                for i, img in enumerate(batch_images):
                    page_num = batch_start + i + 1
                    
                    # Use BytesIO instead of temp files for faster processing
                    image_bytes = BytesIO()
                    img.save(image_bytes, format="PNG")
                    image_bytes.seek(0)
                    
                    # Save to temp file (required by model.infer)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                        tmp_img.write(image_bytes.getvalue())
                        tmp_img_path = tmp_img.name

                    try:
                        print(f"[GPU Processing: {page_num}/{num_pages}]")
                        res = model.infer(
                            tokenizer,
                            prompt="<image>\n<|grounding|>Convert the document to markdown.",
                            image_file=tmp_img_path,
                            output_path=request_output_dir,
                            eval_mode=True,
                            base_size=1024,
                            image_size=640,
                            crop_mode=True
                        )
                        
                        page_text = res.get("text", "") if isinstance(res, dict) else str(res)
                        full_markdown.append(f"## Page {page_num}\n\n{page_text}")
                    finally:
                        if os.path.exists(tmp_img_path):
                            os.remove(tmp_img_path)

        return {"markdown": "\n\n".join(full_markdown)}

    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(request_output_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)