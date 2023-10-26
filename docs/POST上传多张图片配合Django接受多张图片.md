# POST上传多张图片配合Django接受多张图片

**本地**：POST发送文件，使用的是files参数，将本地的图片以二进制的方式发送给服务器。

在这里 files=[("img",open('./2.jpg', 'rb')),("img",open('./1.jpg', 'rb'))]将所有二进制文件放在了img这个键下。

```python
def upload():
    try:
        files=[("img",open('./2.jpg', 'rb')),("img",open('./1.jpg', 'rb'))]
        x = requests.post("http://127.0.0.1:8000/message/1/submmit",files=files)
        print(json.loads(x.text, encoding='utf-8'))
    except Exception as e:
        print(e)
```

**服务端：**使用Django的方法将PSOT请求中的二进制文件读出来.

这样用getlist()方法，将img所对应的多个二进制文件读出。然后以写二进制文件的方式，将每个item写入./media/文件名 中。

```python
imgSrc=request.FILES.getlist('img')
for item in imgSrc:
    with open("./media/"+item.name,'wb') as f:
        for c in item.chunks():
    	    f.write(c)
```

**结果：**

![image.png](https://i.loli.net/2019/11/27/WgQX8UDLr5mjh3I.png)

