持续更新

## 1.Windows下设置GitHub SSH

- gitbash上设置账户,邮箱,密码

  ![image.png](https://i.loli.net/2020/11/08/o5MhGZBFDPHTCn4.png)

  ```
  git config --global user.name "zhangsan(用户名)"
  git config --global user.email "123456@qq.com(邮箱)"
  git config --global user.password "123456(密码)"
  ```

- 设置ssh

  ```
  ssh-keygen -t rsa -C "邮箱"
  ```

  连续三个回车(选择默认设置)

  ![image.png](https://i.loli.net/2020/11/08/gfQrBjlETuyCGJP.png)

  

- 添加密钥到仓库

  打开Setting-SSH and GPG keys选择new  SSH key

  ![image.png](https://i.loli.net/2020/11/08/CBfM4N5KxZgUskq.png)

  打开本地生成的id_rsa.pub文件,内容复制到ssh中.

- 测试是否成功

  打开git-bash 输入

  ```
  ssh -T git@github.com
  ```

  ![image.png](https://i.loli.net/2020/11/08/RyLiuo4N5xJrWdX.png)

## 2.本地项目初始化并提交到远程仓库

```
//1. 初始化本地项目
git init
//2. 添加文件
git add .
//3. 提交文件
git commit -m "First commit"
//4. 添加远程仓库
git remote add origin [仓库地址]
//5. 确认地址
git remote -v
//6. push到远程仓库
git push -u origin master
```