from http import HTTPStatus
from dashscope import Generation
import dashscope
from dashscope.api_entities.dashscope_response import Role


def call_with_messages(content):
    dashscope.api_key = (
        "sk-aa66942234f74452a8a02047fdbc2433"  # 将 YOUR_API_KEY 改成您创建的 API-KEY
    )
    messages = [
        {"role": "system", "content": "recommendation system"},
        {"role": "user", "content": content},
    ]
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_max,
        messages=messages,
        result_format="message",  # 设置结果为消息格式
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]["message"]["content"]
        messages.append(
            {
                "role": response.output.choices[0]["message"]["role"],
                "content": response.output.choices[0]["message"]["content"],
            }
        )
    else:
        print(
            "Request id: %s, Status code: %s, error code: %s, error message: %s"
            % (
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )
        )


if __name__ == "__main__":
    print(
        call_with_messages(
            "Given a user who is 'female, 25-34, and in sales/marketing', this user's movie viewing history over time is listed below: 'What Lies Beneath (2000), 5 star; Ghost (1990), 3 star; Aladdin (1992), 4 star; Toy Story (1995), 5 star; Scream (1996), 5 star; The Silencce of the Lambs(1991), 5 stars'. Analyze user's preferences (consider factors like 'genre, director, actors, time period, country, character, plot/theme, mood/tone, critical acclaim/award'. Provide clear explanations based on details from the user's viewing history and other pertinent factors."
        )
    )
