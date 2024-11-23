from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(title="WikiLLM Inference Service")


# 请求模型
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    num_return_sequences: int = 1


# 响应模型
class GenerationResponse(BaseModel):
    generated_text: str


# 全局变量存储模型和tokenizer
model = None
tokenizer = None


@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        logging.info("正在加载模型和tokenizer...")
        model_path = "./WikiLLM/Weight"

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # 自动选择设备
            torch_dtype=torch.float16  # 使用float16以减少内存占用
        )

        logging.info("模型加载完成!")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise RuntimeError("模型初始化失败")


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="模型未正确加载")

    try:
        # 准备输入
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_return_sequences=request.num_return_sequences,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return GenerationResponse(generated_text=generated_text)

    except Exception as e:
        logging.error(f"生成过程发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": model is not None}


def main():
    logging.info("启动推理服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()