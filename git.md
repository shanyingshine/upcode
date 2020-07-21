# Git基础  
Git 只暂存上一次git add命令时的版本，所以每次修改要git add  
`Untracked files`:新增的文件  
`git status`: 查看状态  
`git add`: 跟踪新文件，把已跟踪的文件放进暂存区，合并时把有冲突的文件标记为已解决状态  
`git status -s`或者`git status -short`:格式更为紧凑的输出  
`??`：新增加的未跟踪文件　　
`A`：新增加到暂存区中的文件  
`M`：修改过的文件  
`MM`:已修改，暂存后又作了修改，因此改文件的修改中既有已暂存的部分，又有未暂存的部分。  
`git diff`:可以知道具体修改了什么地方。`git diff`本身只显示尚未暂存的改动，而不是自上次提交以来所做的所有改动。
`git diff -staged`:查看已暂存的将要添加到下次提交里的内容。  
`git commit`：提交代码  
`git commit -a`:跳过`git add`把所有已经跟踪的代码暂存并提交。  
## 忽略文件
创建一个`.gitignore`的文件，列出要忽略的文件的模式。
```
$ cat.gitignore
*.[oa]
*~
```
第一行告诉 Git 忽略所有以 .o 或 .a 结尾的文件。一般这类对象文件和存档文件都是编译过程中出现的。 第二行告诉 Git 忽略所有名字以波浪符（~）结尾的文件，许多文本编辑软件（比如 Emacs）都用这样的文件名保存副本。 此外，你可能还需要忽略 log，tmp 或者 pid 目录，以及自动生成的文档等等。 要养成一开始就为你的新仓库设置好 .gitignore 文件的习惯，以免将来误提交这类无用的文件。  
文件 .gitignore 的格式规范如下：  
所有空行或者以 # 开头的行都会被 Git 忽略。  
可以使用标准的 glob 模式匹配，它会递归地应用在整个工作区中。  
匹配模式可以以（/）开头防止递归。  
匹配模式可以以（/）结尾指定目录。  
要忽略指定模式以外的文件或目录，可以在模式前加上叹号（!）取反。  
## 移除文件
`git rm`: 从已跟踪文件清单之中移除(从暂存区域移除)，提交  
![](.git_images/a8c3c356.png)
`git rm --cached ` 从git中删除，但是保存在本地磁盘中
## 移动文件
`git mv file_from file_to`改名  
`git log` 查看提交历史  
`git log -p`显示每次提交所引起的差异，可以在后面加上`-2`限制为最近两次提交
`-stat`每次提交的下面列出所有被修改过的文件、有多少文件被修改了，被修改的文件哪些行被移除或是添加  
`-pretty=`提供不同于默认格式的方式展示提交历史,`oneline,short,full,fuller`
`-pretty=format`定制记录的显示格式。
![](.git_images/-pretty=format.png)
![](.git_images/-format-graph.png)
![](.git_images/git-log.png)
`git commit --amend`撤销操作  
`git checkout -- `用最近提交的版本覆盖本地版本  
## 远程仓库的使用
