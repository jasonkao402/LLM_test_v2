from datetime import datetime, timedelta, timezone

TWTZ = timezone(timedelta(hours = 8))
    
def clamp(n:int, minn=0, maxn=100) -> float:
    '''clamp n in set range'''
    return max(min(maxn, n), minn)

def sepLines(itr, sep='\n'):
    return sep.join(itr)

def utctimeFormat(t:datetime):
    return t.replace(tzinfo=timezone.utc).astimezone(TWTZ).strftime("%Y-%m-%d %H:%M:%S")

class replyDict:
    def __init__(self, role: str = 'assistant', content: str = '', name: str = '', image_url: str = ''):
        self.role = role
        self.name = name
        if len(image_url) > 0:
            self.content = [{'type': 'text', 'text': content}, {'type': 'image_url', 'image_url': {'url': image_url, 'detail': "auto"}}]
        else:
            self.content = content

    def __str__(self):
        return f"{self.role} : {self.content}"
    
    @property
    def asdict(self):
        result = {'role': self.role, 'content': self.content}
        if self.name:
            result['name'] = self.name
        return result