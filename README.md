# 梯度提升树——自定义损失函数


----------


典型的梯度提升树（GDBT）有xgboost, lightgbm，其算法的具体细节这里不再描述。根本上，GDBT由一系列CART（分类与回归树）组成：
\begin{equation}\nonumber
\hat y_i = \sum^{K}_{k=1}f_k(x_i) 
\end{equation}
这里的每个$f_i(x)$代表一棵树。

----------

监督学习的目标函数千篇一律，均为通过更新迭代参数$\theta$，使得预测值$\hat y$接近真实值$y$
\begin{equation}\nonumber
obj(\theta) = \sum^{n}_{i=1}l(y_i, \hat y_i) + \sum^{K}_{k=1}\Omega(f_k) 
\end{equation}
这里的第一项是模型预测损失函数，第二项对模型的复杂度进行约束，起正则化的作用

GDBT的第$t$步是在$t-1$步的基础上再建立一棵树$f_t$使得目标函数最小，是一个逐步独立建树的过程，正是因为如此，GDBT往往难以并行训练
\begin{equation}\nonumber
\hat y^{t}_i = \sum^{t}_{k=1}f_k(x_i) = \hat y^{t-1}_{i} + f_t(x_i)
\end{equation}

对于建立$f_t$，通过在$\hat y^{t-1}_i$泰勒展开，其优化目标为
\begin{equation}\nonumber
obj^{t} = \sum^{n}_{i=1}l(y_i, \hat y^{t}_i) = \sum^{n}_{i=1}[l(y_i, \hat y^{t-1}_i) + g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)] + \Omega(f_t) +c
\end{equation}
这里涉及到一阶导数（对应：Jacobi，雅可比矩阵）和二阶导数（对应：Hessian，海森矩阵）：
\begin{equation}\nonumber
g_i = \frac{\partial l(y_i, \hat y^{t-1}_i)}{\partial \hat y^{t-1}_i}\\
h_i = \frac{\partial^2 l(y_i, \hat y^{t-1}_i)}{\partial^2 \hat y^{t-1}_i}
\end{equation}

可见，建立$f_t$，依赖于得到每个数据点下目标函数的一阶导数和二阶导数。

----------

如果损失函数是平方失真$l = (y-\hat{y})^2$

损失函数一阶导数：
 \begin{equation}\nonumber
\frac{\partial L}{\partial \hat y} =2(\hat{y} - y)
\end{equation}

二阶导数：
 \begin{equation}\nonumber
\frac{\partial^2 L}{\partial \hat y^2} = 2
\end{equation}

    def loss(preds, train_data):
    
        labels = train_data.get_label()
        grad = 2*(preds - train_data)
        hess = 2*np.ones(np.size(y_true))
        return  grad, hess

----------

如果是交叉熵损失
 \begin{equation}\nonumber
l = -[ylog(h(\hat{y})) +(1-y)log(1-h(\hat{y}))]
\end{equation}
通常$h(x)$为Sigmoid函数，
 \begin{equation}\nonumber
h(x) = \frac{1}{1+e^{-x}}\\
\end{equation}
其导数：
 \begin{equation}\nonumber
h'(x) = h(x)*(1-h(x))
\end{equation}

则
 \begin{equation}\nonumber
l = -[-ylog(1 + e^{-\hat{y}})) -\hat{y}(1-y) - (1-y)log(1 + e^{-\hat{y}})]\\
= \hat{y}(1-y) + log(1 + e^{-\hat{y}})
\end{equation}

损失函数一阶导数：
 \begin{equation}\nonumber
\frac{\partial L}{\partial \hat y} =h(\hat{y} ) - y
\end{equation}

二阶导数：
 \begin{equation}\nonumber
\frac{\partial^2 L}{\partial \hat y^2} = h(\hat{y} )*(1 -  h(\hat{y} ))
\end{equation}

但是上述表达$\hat y$出现在分母，容易导致分母为0的异常，所以经常对$\hat y$进行映射。

python 实现

    def lgb_loss(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess
