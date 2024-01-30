# "access_token":"24.56c4287d1f448e7d9c5c842fc63a123e.2592000.1706761961.282335-46236564"
import requests
class translate():
    def __init__(self):
        self.token = "24.56c4287d1f448e7d9c5c842fc63a123e.2592000.1706761961.282335-46236565"
        self.url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + self.token
    def english_to_chinese(self, question):
        q = question
        from_lang = 'en'  # example: en
        to_lang = 'zh'  # example: zh
        term_ids = ''  # 术语库id，多个逗号隔开
        headers = {'Content-Type': 'application/json'}
        payload = {'q': q, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()
        return result['result']['trans_result'][0]['dst']

if __name__ == '__main__':
    test = translate()
    result = test.english_to_chinese("hello world")
    print(result)
