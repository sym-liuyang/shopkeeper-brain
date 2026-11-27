"""
    @Author:
    @Time:2026/11/23
    @Desc:
"""
import sys

from pymilvus import DataType

from clients.milvus_utils import get_milvus_client
from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from task_utils import add_running_task, add_done_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    embeddings_content = state.get("embeddings_content")
    if not embeddings_content:
        logger.error(f"embeddings_content没有值,业务无法继续,提前终止!")
        raise ValueError(f"embeddings_content没有值,业务无法继续,提前终止!")

    return embeddings_content

@step_log("step_2_prepared_item_name_collection")
def step_2_prepared_item_name_collection():
    # 1. 创建Milvus 客户端
    client = get_milvus_client()

    # 判断集合是否存在
    has_collection = client.has_collection(collection_name=milvus_config.chunks_collection)
    if has_collection:
        # 如果存在 直接返回
        logger.info(f"{milvus_config.chunks_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.chunks_collection}不存在,进行集合的创建!")

    # 3.1. Create schema
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    # 3.2. Add fields to schema
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # Prepare index building params
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",  # Name of the vector field to be indexed
        index_type="HNSW",  # Type of the index to create
        index_name="denser_index",  # Name of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }
    )

    index_params.add_index(
        field_name="sparse_vector",  # Name of the sparse vector field to index
        index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
        index_name="sparse_index",  # Name of the index to create
        metric_type="IP",  # Metric type used for full text search
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 3.5. Create a collection with the index loaded simultaneously
    client.create_collection(
        collection_name=milvus_config.chunks_collection,
        schema=schema,
        index_params=index_params
    )

    client.load_collection(collection_name=milvus_config.chunks_collection)

@step_log("step_3_insert_item_name_data")
def step_3_insert_item_name_data(embeddings_content):
    # 1. 创建Milvus 客户端
    client = get_milvus_client()

    client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title == '{embeddings_content[0].get("file_title")}'"
    )

    client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )



@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 1. 添加运行时转改
    add_running_task(state.get("task_id"),"node_import_milvus")

    # 2. 校验状态结果
    embeddings_content = step_1_validate_and_get_data(state)

    # 3. 创建Milvus Collection
    step_2_prepared_item_name_collection()

    # 4. 插入数据
    step_3_insert_item_name_data(embeddings_content)

    logger.info(
        f"{embeddings_content[0].get("file_title")}文档对应的数据已经完成向量数据库的导入,导入的数量:{len(embeddings_content)}!")
    add_done_task(state['task_id'], 'node_import_milvus')

    # 添加结束状态
    add_done_task(state.get("task_id"),"node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "file_title": "test.pdf",
        "embeddings_content": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 执行节点函数
        result_state = node_import_milvus(test_state)
    except Exception as e:
        print(f"❌ 测试失败: {e}")