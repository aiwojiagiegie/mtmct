import requests


def send_feishu_message(message: str):
    """发送消息到飞书机器人
    
    Args:
        message: 要发送的消息文本
    
    Returns:
        bool: 是否发送成功
    """
    webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/f3c7b461-8982-4c55-a33e-1b74ccf7ac52"

    data = {
        "msg_type": "text",
        "content": {
            "text": message
        }
    }

    response = requests.post(webhook_url, json=data)
    return response.status_code == 200
