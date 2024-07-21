import re

def postprocessing(text):

        pattern = re.compile(r'https?://\S+|www\.\S+')
        text = pattern.sub(' ', text)

        pattern = re.compile(r'[*#,!?$@:;]')
        text = pattern.sub('', text)

        for pattern in ['The provided text mentions that', 'Based on the provided text', 'According to the provided text', 'The provided text states that', 'The provided text', 'The text you provided lists', 'The text you provided', 'The information you provided']:
            text = re.sub(pattern, '' , text)

        pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        text = pattern.sub(' ', text)

        return text.strip()