# EvalScope  

[EvalScope](https://github.com/modelscope/evalscope) is the official model evaluation and benchmarking framework for [ModelScope](https://modelscope.cn/), designed to meet diverse evaluation needs. It supports various model types including large language models (LLMs), multimodal models, embedding models, reranker models, and CLIP models.  

![evalscope-logo](./evalscope/logo.png)  

The framework accommodates multiple evaluation scenarios such as end-to-end RAG evaluation, arena mode, and inference performance testing. It comes pre-loaded with benchmarks and metrics including MMLU, CMMLU, C-Eval, and GSM8K. Seamlessly integrated with the [ms-swift](https://github.com/modelscope/ms-swift) training framework, EvalScope enables one-click evaluation, providing comprehensive support for model training and evaluation 🚀.  

Now you can use EvalScope to evaluate LLM performance while leveraging SwanLab for convenient tracking, comparison, and visualization.  

[Demo](https://swanlab.cn/@ShaohonChen/perf_benchmark/overview)

## 1. Preparation  

Install the required environment:  

```bash  
pip install evalscope  
pip install swanlab  
```  

For extended EvalScope functionality, install optional dependencies as needed:  

```bash  
pip install -e '.[opencompass]'   # Install OpenCompass backend  
pip install -e '.[vlmeval]'       # Install VLMEvalKit backend  
pip install -e '.[rag]'           # Install RAGEval backend  
pip install -e '.[perf]'          # Install performance dependencies  
pip install -e '.[app]'           # Install visualization dependencies  
pip install -e '.[all]'           # Install all backends (Native, OpenCompass, VLMEvalKit, RAGEval)  
```  

## 2. Evaluating Qwen Model Performance

If you want to evaluate ` Qwen2.5-0.5B-Instruct ` on the [default data in openqa format](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese) while monitoring results via SwanLab, run the following command: 

```bash  {5,6}
export CUDA_VISIBLE_DEVICES=0
evalscope perf \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --dataset openqa \
 --number 20 \
 --parallel 2 \
 --limit 5 \
 --swanlab-api-key 'your API Key' \
 --name 'qwen2.5-openqa' \
 --temperature 0.9 \
 --api local 
```  

Where:  
• `swanlab-api-key` is your SwanLab API key  
• `name` specifies the experiment name  

To customize the project name, navigate to the `statistic_benchmark_metric_worker` function in `evalscope/perf/benchmark.py` and modify the `project` parameter in the SwanLab configuration section.  

**Visualization Effect Example:**

![](./evalscope/show.png)

## Upload to Self-Hosted Version  

If you wish to upload the evaluation results to a self-hosted version, you can first log in to the self-hosted version via the command line. For example, if your deployment address is `http://localhost:8000`, you can run:  

```bash  
swanlab login --host http://localhost:8000  
```  

After completing the login, run the `evalscope` command, and the evaluation results will be uploaded to the self-hosted version.