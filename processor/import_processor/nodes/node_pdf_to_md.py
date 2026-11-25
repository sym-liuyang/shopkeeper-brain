"""
    @Author:
    @Time:2026/11/23
    @Desc:
"""
import os
import shutil
import time
from pathlib import Path

import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from path_util import PROJECT_ROOT
from processor.import_processor.state import ImportGraphState, create_default_state
from task_utils import add_done_task, add_running_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取pdf_path 和 local_dir
    # 判断是否为空
    # 如果 pdf_path 为空 抛出异常， 如果local_dir为空，指定默认输出目录
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    if not pdf_path:
        logger.error("状态中pdf_path为空,流程终止")
        raise ValueError("pdf_path为空")

    if not local_dir:
        logger.error("状态中local_dir为空,指定默认路径")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = local_dir

    # 转换为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 判断文件是否存在
    if not pdf_path_obj.is_file():
        # 如果pdf_path 文件不存在，抛出异常
        logger.error("pdf_path 文件不存在或者不是文件,流程终止")
        raise ValueError("pdf_path 文件不存在或者不是文件")

    if not local_dir_obj.is_dir():
        # 如果local_dir_path 文件不存在，创建新目录
        logger.warning("local_dir_path 文件不存在,创建新目录")
        # parents=True 表示创建父目录，如果父目录不存在则一并创建
        # exist_ok=True 表示如果目录已存在则不会抛出异常
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    # 0. 校验MinerU配置
    if not mineru_config.api_key or not mineru_config.base_url:
        logger.error("MinerU 配置信息有误，请校验！")
        raise ValueError("MinerU 配置信息有误，请校验！")

    # 1. 申请上传地址
    token = mineru_config.api_key
    url = mineru_config.base_url + "/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        # "model_version": "pipeline"
        "model_version": "vlm"
    }

    response = requests.post(url, headers=header, json=data)
    http_status_code = response.status_code
    if http_status_code != 200:
        logger.error("向MinerU申请上传地址失败，状态码：{}".format(http_status_code))
        raise RuntimeError("向MinerU申请上传地址失败")

    # 获取相应的内容
    result = response.json()
    # 获取业务接口返回的状态码
    response_code = result.get("code")
    # 获取业务接口返回的描述消息
    response_msg = result.get("msg")
    if response_code != 0:
        logger.error("向MinerU申请上传地址失败，状态码：{}, 消息：{}".format(response_code, response_msg))
        raise RuntimeError("向MinerU申请上传地址失败，状态码：{}, 消息：{}".format(response_code, response_msg))

    batch_id = result.get("data").get("batch_id")
    file_urls = result.get("data").get("file_urls")

    if not batch_id:
        logger.error("向MinerU申请上传地址失败，batch_id获取失败")
        raise RuntimeError("向MinerU申请上传地址失败，batch_id获取失败")

    if not file_urls:
        logger.error("向MinerU申请上传地址失败，file_urls获取失败")
        raise RuntimeError("向MinerU申请上传地址失败，file_urls获取失败")

    upload_file_url = file_urls[0]

    # 2. 上传文件
    # 向指定的位置上传文件
    file_upload_data = pdf_path_obj.read_bytes()
    # 如果有代理,会在请求头中携带额外的信息,导致存储文件的服务器不能访问而失败
    # res_upload = requests.put(upload_file_url, data=file_upload_data)
    with requests.session() as session:
        session.trust_env = False
        session.headers.pop("Content-Type", None)
        res_upload = session.put(upload_file_url, data=file_upload_data)
        if res_upload.status_code != 200:
            logger.error("向MinerU上传文件失败，状态码：{},请检查minerU配置".format(res_upload.status_code))
            raise RuntimeError("向MinerU上传文件失败，状态码：{},请检查minerU配置".format(res_upload.status_code))

    logger.info(">>>> 向{}上传{}文件完成 <<<<".format(upload_file_url, pdf_path_obj))

    # 3. 轮询获取结果
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    # 超市事件
    timeout = 600
    # 每隔多久轮询一次
    interval = 3

    start_time = time.time()
    poll_res = requests.get(poll_url, headers=header)

    while True:
        if time.time() - start_time > timeout:
            logger.error("轮询获取解析结果超时")
            raise RuntimeError("轮询获取解析结果超时")

        try:
            poll_res = requests.get(poll_url, headers=header)
        except:
            logger.error("轮询获取结果失败! 可以稍后重试")
            time.sleep(interval)
            continue

        # 获取请求的状态码
        poll_status_code = poll_res.status_code
        # 判断是不是服务器内部的问题
        if poll_status_code != 200:
            if 500 <= poll_status_code < 600:
                logger.warning("可修复的网络异常, 状态码：{}".format(poll_status_code))
                time.sleep(interval)
                continue
            else:
                logger.error("不可修复的网络异常, 状态码：{}".format(poll_status_code))
                raise RuntimeError("不可修复的网络异常, 状态码：{}".format(poll_status_code))
        # 判断业务接口返回的内容
        poll_result = poll_res.json()
        # 获取业务响应码
        poll_response_code = poll_result.get("code")

        # 判断业务接口返回的状态码
        if poll_response_code != 0:
            logger.error("获取解析结果失败，状态码：{}, 消息：{}".format(poll_result.get("code"), poll_result.get("msg")))
            raise RuntimeError(
                "获取解析结果失败，状态码：{}, 消息：{}".format(poll_result.get("code"), poll_result.get("msg")))

        # 获取业务响应消息
        poll_response_data = poll_result.get("data")
        extract_data = poll_response_data.get("extract_result")[0]
        extract_data_state = extract_data.get("state")

        if extract_data_state == "done":
            extract_data_url = extract_data.get("full_zip_url")
            if not extract_data_url:
                logger.error("获取解析结果失败，full_zip_url获取失败")
                raise RuntimeError("获取解析结果失败，full_zip_url获取失败")
            logger.info(">>>> 获取解析结果成功, full_zip_url: {} <<<<".format(extract_data_url))
            return extract_data_url
        elif extract_data_state == "failed":
            logger.error("获取解析结果失败， 消息：{}".format(extract_data.get("err_msg")))
            raise RuntimeError("获取解析结果失败， 消息：{}".format(extract_data.get("err_msg")))
        else:
            logger.warning("解析正在进行中,状态：{}, 请稍后。。。".format(extract_data_state))
            time.sleep(interval)
            continue

@step_log("step_3_download_and_unzip")
def step_3_download_and_unzip(zip_url, local_dir_path, stem):
    # 从服务器下载文件
    response = requests.get(zip_url, timeout=60)
    if response.status_code != 200:
        logger.error("从服务器下载文件失败，状态码：{}".format(response.status_code))
        raise RuntimeError("从服务器下载文件失败，状态码：{}".format(response.status_code))

    md_zip_path_obj: Path = local_dir_path / f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(response.content)

    # 解压到指定目录
    extract_path_obj = local_dir_path / stem
    if extract_path_obj.is_dir():
        # 如果已经存在，先删除
        logger.warning("解压目录已存在，先删除")
        shutil.rmtree(extract_path_obj)

    extract_path_obj.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(md_zip_path_obj, extract_path_obj)

    # 查找解压后的md文档
    extract_list = list(extract_path_obj.rglob("*.md"))
    if not extract_list:
        logger.error("解压后{}没有找到md文件".format(extract_path_obj))
        raise RuntimeError("解压后{}没有找到md文件".format(extract_path_obj))

    # 按照优先级选择md文件 stem.md   full.md   list[0]
    for md_file_obj in extract_list:
        if md_file_obj.stem == stem:
            return md_file_obj

    target_file_obj = None
    for md_file_obj in extract_list:
        if md_file_obj.name.lower() == "full.md":
            target_file_obj = md_file_obj
            logger.info(">>>> 获取解析结果成功, 名字是full.md, 后续需要改名 <<<<")
        break

    if not target_file_obj:
        target_file_obj = extract_list[0]

    # 改名为 stem.md
    final_md_path_pbj = target_file_obj.rename(target_file_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{md_zip_path_obj}解压成功，目标文件{target_file_obj}")

    return final_md_path_pbj


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 1. 添加任务运行状态
    add_running_task(state.get("task_id"), "node_pdf_to_md")

    # 2. 从状态中获取数据并进行校验
    pdf_path_obj, local_dir_path = step_1_validate_and_get_data(state)

    # 3. 上传文件并解析
    zip_url = step_2_upload_and_poll(pdf_path_obj)

    # 4. 下载并解压
    md_path = step_3_download_and_unzip(zip_url, local_dir_path, pdf_path_obj.stem)
    state["md_path"] = md_path

    # 5. 添加任务
    add_done_task(state.get("task_id"), "node_pdf_to_md")
    return state


if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)
    print(test_state)
    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
