## 1. 安装NodeJs

首先下载NodeJs的安装包，下载地址：https://nodejs.org/dist/v12.14.0/node-v12.14.0-x64.msi

安装完成后设置环境变量：

```
# 修改npm的全局安装模块路径
npm config set prefix "D:\developer\env\node-v12.14.0\node_global"
# 修改npm的缓存路径
npm config set cache "D:\developer\env\node-v12.14.0\node_cache"
```

打开系统环境变量设置：
添加系统环境变量NODE_PATH；设置value: D:\developer\env\node-v12.14.0

在系统环境变量的Path中添加如下路径:

```
%NODE_PATH%\
%NODE_PATH%\node_global\
```

## 2. 安装Docsify

```
npm i docsify-cli -g
```

## 3. 启动docs
进入monarch-preprocess-app\docs\beta>目录。执行：docsify serve docs

## 4. 预览
浏览器打开http://localhost:3000