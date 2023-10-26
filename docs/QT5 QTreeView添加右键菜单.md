# C++ QT5学习——QTreeView控件创建右键菜单

QTreeView是QWidget的子类，我们再改写QTreeView类的时候，注意的是继承关系。

## 1.**TreeView.h**

```c++
class TreeView : public QTreeView//记得加public 不然是私有继承
{
    Q_OBJECT //使用信号与槽所必需的
    public:
        TreeView();   
    public slots:
        void slotCustomContextMenu(const QPoint &point);//创建右键菜单的槽函数
};
```

切入正题。

对于QTreeView实现右键菜单是通过信号与槽实现的。

我们在点击右键的时候会发生customContextMenuRequested(const QPoint &)信号。我们根据这个信号创建菜单就行了

## 2.TreeView.cpp

```c++
TreeView::TreeView() :QTreeView() //构造函数
{

    this->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this,SIGNAL(customContextMenuRequested(const QPoint &)),this, SLOT(slotCustomContextMenu(const QPoint &)));
}

void TreeView::slotCustomContextMenu(const QPoint &point) //槽函数定义
{
        QMenu *menu = new QMenu(this);
        QAction *a1=new QAction(tr("上传"));
        menu->addAction(a1);
        QAction *a2=new QAction(tr("移动"));
        menu->addAction(a2);
        QAction *a3=new QAction(tr("复制"));
        menu->addAction(a3);
        QAction *a4=new QAction(tr("删除"));
        menu->addAction(a4);
        menu->exec(this->mapToGlobal(point));

}
```

这样就实现了右键的菜单显示

## 3.效果显示

![捕获.PNG](https://i.loli.net/2019/10/05/mlSvxa1MjtgJN2p.png)