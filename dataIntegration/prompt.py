"""
生成初始的train、val、test
"""

from openai import OpenAI

# use your api_key here
client = OpenAI(
    api_key="",
    base_url="https://api.chatanywhere.tech/v1",
)


# TODO: The 'openai.base_url' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://api.chatanywhere.tech/v1")'
# openai.base_url = "https://api.chatanywhere.tech/v1"
# MODEL_NAME = "gpt-4"
MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "gpt-4o"


def test_openai_api(question):
    sign = True
    while sign:
        try:
            rsp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": question},
                ],
            )
            sign = False
        except:
            sign = True
    return rsp.choices[0].message.content


if __name__ == "__main__":
    prompt = "viewing history: (Christmas Story, A, 5 star); (Star Wars: Episode IV - A New Hope, 4 star); (Wallace & Gromit: The Best of Aardman Animation, 3 star); (One Flew Over the Cuckoo's Nest, 5 star); (Wizard of Oz, 4 star); (Fargo, 4 star); (Run Lola Run, 4 star); (Rain Man, 5 star); (Saving Private Ryan, 5 star); (Awakenings, 5 star).\nPlease analyze the user's preferences in 60 words based on the viewing history."

    result = test_openai_api(prompt)
    print(result)
