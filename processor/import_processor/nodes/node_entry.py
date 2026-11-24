"""
    @Author:
    @Time:2026/11/23
    @Desc:
"""
import os.path
import sys
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState
from task_utils import add_done_task
from utils.task_utils import add_running_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """
    add_running_task(state.get("task_id"), "node_entry")

    local_file_path = state.get("local_file_path")

    if not local_file_path:
        logger.error("Local file path is not provided.")
        return state

    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.error("Unsupported file type: %s", Path(local_file_path).suffix)
        return state

    # state["file_title"] = os.path.basename(local_file_path).split(".")[0]
    # .name 带扩展名  .stem 不带扩展名  .parent  父目录  .suffix 获取后缀
    state["file_title"] = Path(local_file_path).stem

    add_done_task(state.get("task_id"), "node_entry")
    return state
