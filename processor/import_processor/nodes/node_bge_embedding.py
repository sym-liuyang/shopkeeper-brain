"""
    @Author:
    @Time:2026/11/23
    @Desc:
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from common.logging.logger import logger, node_log, step_log
from lm.embedding_utils import generate_embeddings
from processor.import_processor.state import ImportGraphState
from task_utils import add_running_task, add_done_task

# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取chunks
    chunks = state.get("chunks")

    if not chunks:
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks: list[dict[str, Any]] = json.loads(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk["item_name"] = state.get("item_name")
                state["chunks"] = chunks
        else:
            logger.error(f"chunks没有值,json备份文件不存在,抛出异常!")
            raise ValueError(f"chunks没有值,json备份文件不存在,抛出异常!")

    return chunks


@step_log("step_2_batch_generate_vector")
def step_2_batch_generate_vector(chunks):
    # 分批次取chunk
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        current_chunks = chunks[i: i + EMBEDDING_BATCH_SIZE]
        content_list: list[str] = [f"{chunk.get("item_name")}_{chunk.get("content")}" for chunk in current_chunks]
        # 调用嵌入模型，给列表中每一个元数生成向量
        result = generate_embeddings(content_list)
        dense_list = result["dense"]
        sparse_list = result["sparse"]
        for index, chunk in enumerate(current_chunks):
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]

    logger.info(f"chunks已经批量生成向量! {chunks[:1]}")


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 1. 添加运行时状态
    add_running_task(state.get("task_id"), "node_bge_embedding")

    # 2. 从状态中获取数据校验
    chunks = step_1_validate_and_get_data(state)

    # 3. 对每个chunk进行向量化
    step_2_batch_generate_vector(chunks)

    # 4. 更新状态
    state["embeddings_content"] = chunks

    # 5.添加结束状态
    add_done_task(state.get("task_id"), "node_bge_embedding")
    return state


if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果:{result_chunks}")


    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")
