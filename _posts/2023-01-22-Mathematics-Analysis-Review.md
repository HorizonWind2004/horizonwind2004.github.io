---
title: 'Mathematics Analysis Review'
date: 2023-01-22
permalink: /posts/2023/01/mathematics-analysis-review/
tags:
  - course
use_math: true

---
题源大概是答疑群里同学的野题，课本，真题卷，PICARD Fluid Dynamic 学长（万分感谢 orz）的讲义，以及我自己找到的一些野题。


------------

> 设 $f,g$ 为 $D$ 上的有界函数，证明：

> $$\inf\{f(x)+g(x)\}\le\inf f(x)+\sup g(x)$$

利用常数放缩。

> $$\inf\{f(x)+g(x)\}\le\inf\{f(x)+\sup g(x)\}\le\inf f(x)+\sup g(x)$$

同样也可以证明：

> $$\sup f(x)+\inf g(x)\le\sup\{f(x)+g(x)\}$$

------------

> 设 $\lim\limits_{n\to\infty}a_n=a,|\lambda|<1$，请用 $\varepsilon-N$ 语言证明：

> $$\lim\limits_{n\to\infty}\{a_n+a_{n-1}\lambda+...+a_0\lambda^n\}=\frac{a}{1-\lambda}$$

直接做太坐牢了，还有正负号讨论，根本做不了。

构造新的数列：

$$a+a\lambda+...+a\lambda^n$$

那么构造数列 $\{b_n=a_n-a\}$ ，考虑用极限减法转化成简单些的问题：

> 设 $\lim\limits_{n\to\infty}b_n=0,|\lambda|<1$，请用 $\varepsilon-N$ 语言证明：

> $$\lim\limits_{n\to\infty}\{b_n+b_{n-1}\lambda+...+b_0\lambda^n\}=0$$

我们下面证明这个数列的绝对值收敛于 $0$，显然这是等价的。

对于一个需要证明的 $\varepsilon>0$：

根据极限的性质，设 $n>N_1$ 时 $|b_n|<\varepsilon(1-|\lambda|)$。所以当 $n>N_1$ 时：

$$|b_n|+|b_{n-1}||\lambda|+...+|b_0||\lambda|^n$$

$$<\varepsilon(1-|\lambda|^{n-N_1})+|b_{N_1}||\lambda^{n-N_1}|+...+|b_1||\lambda^{n-1}|+|b_0||\lambda^n|$$

$$<\varepsilon(1-|\lambda|^{n-N_1})+|b_{N_1}||\lambda^{n-N_1}|+...+|b_1||\lambda^{n-N_1}|+|b_0||\lambda^{n-N_1}|$$

若 $|b_0|+|b_1|+...+|b_{N_1}|=0$ 显然成立，否则根据极限的性质，设 $n>N_2$ 时 $|\lambda^{n-N_1}|<\frac{\varepsilon|\lambda|^{n-N_1}}{|b_0|+|b_1|+...+|b_{N_1}|}$。所以当 $n>\max(N_1,N_2)$ 时原式：

$$<\varepsilon(1-|\lambda|^{n-N_1})+\varepsilon|\lambda|^{n-N_1}=\varepsilon$$

得证。

------------

> 拉马努金（Ramanujan）问题：

> $$\sqrt{1+2\sqrt{1+3\sqrt{1+4\sqrt{\dots}}}}$$

这个等于 $3$。一般性的：

$$n=\sqrt{1+(n-1)(n+1)}$$

$$=\sqrt{1+(n-1)\sqrt{1+n(n+2)}}$$

$$=\sqrt{1+(n-1)\sqrt{1+n\sqrt{1+(n+1)(n+3)}}}$$

$$\dots$$

------------

> 证明： 

> $$\sqrt{1+\sqrt{2+\sqrt{3+...+\sqrt{n}}}}$$

有正的上界 $\sqrt[]{5}$。

神秘，因为把根号五展开就是

$$\sqrt{1+\sqrt{16}}$$

$$\sqrt{1+\sqrt{2+\sqrt{196}}}$$

$$\sqrt{1+\sqrt{2+\sqrt{3+193}}}$$

归纳就可以了，真的牛。

一个放缩的方法：

$$\sqrt{\frac{2}{2}}\sqrt{1+\sqrt{2+\sqrt{3+...+\sqrt{n}}}}$$

$$\sqrt{2}\sqrt{\frac{1}{2}+\sqrt{\frac{2}{2^{2^1}}+\sqrt{\frac{3}{2^{2^2}}+...+\sqrt{\frac{n}{2^{2^{n-1}}}}}}}$$

显然有 $2^{n-1}\ge n(n\ge1),\therefore 2^{2^{n-1}-1}\ge n\ (n\ge1)$

那么原式有上界

$$\sqrt{2}\sqrt{\frac{1}{2}+\sqrt{\frac{1}{2}+\sqrt{\frac{1}{2}+...+\sqrt{\frac{1}{2}}}}}<\frac{1+\sqrt 3}{\sqrt 2}<2$$

连根式还很深，不知道以后有没有机会去研究了。

------------

> 证明 $f(x)=x^a \sin x(a\in[-1,0])$ 在 $(0,+\infty)$ 上一致连续。

使用结论！显然这时候 $\lim\limits_{x\to+\infty}f(x)=0,$ 根据习题结论：

> **命题**：若 $f(x)$ 在 $[A,+\infty)$ 上连续，$\lim\limits_{x\to+\infty}f(x)$ 存在，则 $f(x)$ 在 $[A,+\infty)$ 上一致连续。

证明就是用 Cauchy 收敛准则，然后用 cantor 定理 + 一致连续函数的拼接。

那么取 $A=0$，补充定义 $f(0)=\lim\limits_{x\to0^+}f(x)$，因为这时候也显然右极限存在。那么这道题，就证完了捏。

------------

> $f(x)$ 在 $(0,+\infty)$ 上一致连续，$g(x)$ 在 $[0,+\infty)$ 上连续，$\lim\limits_{x\rightarrow+\infty}[f(x)-g(x)]=0$。求证 $g(x)$ 也在 $[0,+\infty)$ 上一致连续。

其实挺简单的。直接上极限的定义。对一个需要证明的 $\varepsilon>0$：

设 $x>X$ 时 $|f(x)-g(x)|<\frac 14\varepsilon$ ，而且根据一致连续的定义, $|x_1-x_2|<\delta_1,|f(x_1)-f(x_2)|<\frac 12\varepsilon,|g(x_1)-g(x_2)|\leq|f(x_1)-f(x_2)|+|f(x_1)-g(x_1)|+|f(x_2)-g(x_2)|<\varepsilon$ 。所以 $g(x)$ 在 $[X,+\infty]$ 上一致连续。

再根据 cantor 定理，$g(x)$ 在 $[0,X]$ 上一致连续。现在我们只需要证明两个闭区间的一致连续函数的并是一致连续函数！也就是

> **命题**：$f(x)$ 在 $[A,B]$ 上是一致连续函数,在 $[B,C]$ 上是一致连续函数，证明在 $f(x)$ 在 $[A,C]$ 上一致连续。

这个证明很 trival，直接取一个 $\varepsilon$，然后 $x_1<x_2\leq B$ 说明一下，$B<x_1<x_2$ 说明一下，$x_1\leq B< x_2$ 用绝对值不等式说明一下即可。



------------


> 证明 $f(x)=x^a \sin x(a>0)$ 在 $(0,+\infty)$ 上不一致连续。

取子列。

$$x_n=2n\pi+(\frac 1n)^a,y_n=2n\pi$$

$$(2n\pi+\frac 1{n^a})^a \sin \frac 1{n^a}$$

$$=\frac{1}{n^a}(2n\pi+\frac 1{n^a})^a \frac{\sin \frac 1{n^a}}{\frac 1{n^a}}$$

$$=(2\pi+\frac 1{n^{a+1}})^a \frac{\sin \frac 1{n^a}}{\frac 1{n^a}}$$

$$\lim_{n\rightarrow\infty}(2\pi+\frac 1{n^{a+1}})^a \frac{\sin \frac 1{n^a}}{\frac 1{n^a}}=(2\pi)^a$$	

得证。

------------


> 证明 $f(x)=x^a \sin x(a<-1)$ 在 $(0,+\infty)$ 上不一致连续。

一样的取子列。只不过要趋向 $0$ 了。那就更简单了。

$$f(x)=x^{a+1} \frac{\sin x}{x}$$

$$x_n=n^{\frac1{a+1}},y_n=(1+n)^{\frac1{a+1}}$$

$$f(y_n)-f(x_n)=(1+n)\frac{\sin y_n}{y_n}-n\frac{\sin x_n}{x_n}$$

$$x_n>y_n,\frac{\sin y_n}{y_n}>\frac{\sin x_n}{x_n}$$

$$f(y_n)-f(x_n)\ge(1+n)\frac{\sin x_n}{x_n}-n\frac{\sin x_n}{x_n}=\frac{\sin x_n}{x_n}$$

由极限的保序性，

$$\lim_{n\rightarrow\infty}[f(y_n)-f(x_n)]\ge 1$$

得证。

upd：和群友交流，发现有更简单的做法，你取子区间 $(0,1)$ 发现左极限不存在，根据 cantor 定理的推论，开区间连续的充要条件，爆杀。

------------


> 设 $f(x)$ 三阶可导，$f(0) = f'(0) = f''(0) = 0$，$f'''(0) > 0$，$x \in (0, 1)$ 时 $f(x)\in(0,1)$，且数列 $\{x_n\}$ 满足 $x_{n+1}=x_n(1-f(x_n))$。

> 1. 证明： 

> $$\lim_{n\to+\infty}x_n = 0$$

> 2. 证明：存在 $\alpha>0$ 和常数 $c\ne0$，使 

> $$\lim_{n\to+\infty}cn^\alpha x_n = 1$$

第一问：显然 $x_n$ 单调递减。那么等式两边取极限有 $[1-f(\lim\limits_{n\to+\infty}x_n)]\lim\limits_{n\to+\infty}x_n=\lim\limits_{n\to+\infty}x_n$。如果极限不为 $0$ 有 $f(\lim\limits_{n\to+\infty}x_n)=0$。显然 $f$ 是连续的，而值域里没有 $0$，那么极限只能是端点 $0$ 或 $1$。这个显然不可能是 $1$，因为递减。证毕。

没想出第二问，我太菜了。其实这很高中数列，鬼知道为啥我没想出来。

**首先我们得把 $\alpha$ 确定下来**。我也不知道为啥这一步都没想到。其实就是让我们找 $\{x_n\}$ 的增长速度。显然根据泰勒展开，$f(x)$ 在 $0$ 处的增长速度是 $x^3$ 级别的。直接根据高中数列套路（bushi），设 $x_n\sim n^\beta$，有 $\beta-1=4\beta,\beta=-\frac 13$。那么我们取 $\alpha=\frac 13$ 就可以了。这些都写在草稿纸上，我们下面证明：

$$\lim_{n\to+\infty}n^\frac 13 x_n = C$$

也就是

$$\lim_{n\to+\infty}n x_n^3 = C'$$

使用 stolz 定理。那么只需要证明 $\frac 1{x_{n+1}^3}-\frac 1{x_{n}^3}$ 存在极限。直接代入原式：

$$\lim\limits_{n\to+\infty}(\frac 1{x_{n+1}^3}-\frac 1{x_{n}^3})=\lim\limits_{n\to+\infty}\frac {x_{n}^3-x_{n+1}^3}{x_{n}^3x_{n+1}^3}=\lim\limits_{n\to+\infty}\frac {x_{n}^3-x_n^3(1-f(x_n))^3}{x_{n}^3x_{n+1}^3}$$

$$=\lim\limits_{n\to+\infty}\frac {1-(1-f(x_n))^3}{x_{n+1}^3}=\lim\limits_{n\to+\infty}\frac {3f(x_n)-3f(x_n)^2-f(x_n)^3}{x_{n+1}^3}$$

显然 $\lim\limits_{n\to+\infty}\frac {x_{n+1}}{x_{n}}=1-f(0)=1$。

设 

$$f(x_n)=\frac{f'''(\xi_n)}{6}x_n^3$$

那么其实就结束了。原式

$$=\lim\limits_{n\to+\infty}\frac {1-(1-f(x_n))^3}{x_{n+1}^3}=\lim\limits_{n\to+\infty}\frac {3\frac{f'''(\xi_n)}{6}x_n^3-3\frac{f'''(\xi_n)^2}{36}x_n^6-\frac{f'''(\xi_n)^2}{216}x_n^9}{x_{n+1}^3}=\lim\limits_{n\to+\infty}\frac{f'''(\xi_n)}{2}=\frac {f'''(0)}2$$

最后一步用了单侧导数极限定理。

也就是：

$$\lim_{n\to+\infty}\sqrt[3]{\frac {f'''(0)}2}n^{\frac 13}x_n = 1$$


------------

> **海涅定理的推论**：$\lim\limits_{x\to x_0}f(x)$ 存在的充要条件是，任何收敛于 $x_0$ 的数列（$x_n\ne x_0$）$\{x_n\}$，函数值数列 $\{f(x_n)\}$ 一定收敛。（可以推广到单侧极限，趋向无穷）

其实是海涅定理充分性的推广，也就是说，如果所有子列函数值都收敛那么一定收敛于一值。证明就反证法，取出两个子列拼起来。

cantor 定理的证明：反证法，因为闭区间所以有界，任取两个子列，函数值极限不一样，然后 BW 定理得到收敛子列，发现矛盾。

> **cantor 定理的推论**：函数在**有限**开区间上连续，则其在开区间上一致连续的充要条件是函数在端点左右极限存在，即 $f(a+),f(b-)$ 存在。

充分性很简单，补充定义 + cantor 定理即可。关键是必要性。这个其实不常见也不容易想到。

其实还是那个有趣的事情：一致连续把柯西列映射成柯西列。一个趋向于端点的数列一定是柯西列，证明很简单，对于一个 $\varepsilon>0$，根据一致连续定义有一个 $\delta_2$，而又一定存在一个 $\delta_1$ 使得数列 $\{x_n\}$ 中两两绝对值差 $<\delta_2$。那么函数值数列 $\{f(x_n)\}$ 也是收敛的。用海涅定理的推论就证毕了。


------------

> 设 $f: [0,1]\rightarrow[0,1]$ 是连续函数，$f(0)=0,f(1)=1$，并且 $\forall x\in[0,1]$，有 $f(f(x))=x$。证明 $f$ 在 $[0,1]$ 上 **严格**单调递增，并由此进一步证明，$\forall x\in[0,1]$，有 $f(x)=x$。

（第一反应是可以取反函数，这个需要第一问的证明。我们先证第一问。）

考虑反证法。假设存在 $a<b,f(a)\ge f(b)$，而对于任意的 $x_1<x_2(x_1,x_2\in [0,1])$，若 $f(x_1)=f(x_2),f(f(x_1))=f(f(x_2))\Rightarrow x_1=x_2$，矛盾！

所以 $x_1\ne x_2\Rightarrow f(x_1)\ne f(x_2)$ $(*)$。

那么如果连续函数不单调，肯定会有相等吧！这就是我们的切入口。

$\because f(a)\ge f(b)$，那么 $f(0)=0\le f(b)\le f(a)$，根据中间值定理，$[0,a]$ 之间必有 $\xi$ 使得 $f(\xi)=f(b)$。和 $*$ 式矛盾！所以得证。

第二问就简单了。假设存在 $f(x)>x,$ 那么根据单调性 $x=f(f(x))>f(x)$，直接矛盾。$f(x)<x$，也是同理的。

附：根据反函数存在性定理，严格单调说明 $f^{-1}$ 存在，这就引出了一个简单的命题：

> **命题**：设 $f(x): [a,b]\rightarrow[a,b]$ ，$f(x)=f^{-1}(x)\Rightarrow f(x)=x$。


------------

> **单侧导数极限定理**：$f$ 在 $a$ 的半边空心邻域 $U^o_+$ 可导，在半边实心邻域 $U_+$ 连续，如果 $\lim\limits_{x\to a^+}f'(x) = l$ 有限，那么 $f$ 在 $a$ 处右导数存在且等于 $l$。

这个定理用来证明关于补充定义函数的导数存在非常好用。

单侧导数极限定理的严谨证明：

由于在半边实心邻域连续，根据拉格朗日中值定理：

$$\frac{f(x)-f(a)}{x-a}=f'(\xi),\xi\in(a,x)$$

不妨设 $\lim\limits_{x\to a}f'(x)=G.$

对于 $\forall \varepsilon>0,\exists \delta>0,\text{s.t. }\forall x\in(a,a+\delta),|f'(x)-G|<\varepsilon$。则此时对于同样的 $\varepsilon,\delta,\forall x\in(a,a+\delta)$ 也有

$$|\frac{f(x)-f(a)}{x-a}-G|=|f'(\xi)-G|<\varepsilon$$

得证。


------------

> **Leibniz 公式：**

> $$[f(x)g(x)]^{(n)}=\sum_{i=0}^n\binom{n}{i}f(x)^{(i)}g(x)^{(n-i)}$$

Leibniz 公式求函数的 $n$ 阶导数大致有两种思路，第一种是其中一个乘积写几项就成 $0$ 了；第二种是对等式求导，写出递推式。

> 求 $d_n=[\arctan(x)]^{(n)}(0)$。

显然 $d_1=1$。接下来就不是很好做了？这时候我们需要构造一个乘积式：

$$(1+x^2)y^{(1)}=1$$

当 $n\ge 2$ 时，两边求 $n$ 阶导：

$$(1+x^2)d_{n+1}+2nxd_n+n(n-1)d_{n-1}=0$$

$$d_{n+1}+n(n-1)d_{n-1}=0$$

$d_2=0,\therefore$

$$d_n=(-1)^{\frac{n-1}{2}}(n-1)!\frac {1-(-1)^n}2$$

> 求 $d_n=[\arcsin(x)]^{(n)}(0)$。

显然 $d_1=1$。

$$\sqrt{1-x^2}y^{(1)}=1$$

这样子还是没法做，算出来答案会丑到爆炸，因为 $\sqrt{1-x^2}$ 的导数没有特殊性质。所以考虑再求一次导：

$$y^{(2)}=\frac{x}{(1-x^2)^{\frac 32}}$$

神秘操作来了，原式等于

$$y^{(2)}=\frac{xy^{(1)}}{(1-x^2)}$$

这样我们成功的把根号给化掉了！

故技重施：

$$y^{(2)}{(1-x^2)}=xy^{(1)}$$

求 $n(n\ge 2)$ 阶导：

$$y^{(n+2)}{(1-x^2)}-2xy^{(n+1)}-n(n-1)y^{(n)}=xy^{(n+1)}+ny^{(n)}$$

$$d_{n+2}-n(n-1)d_{n}=nd_{n}$$

$$d_{n+2}=n^2d_{n}$$

所以 $d_2=0,d_1=1$

$$d_n={(n-2)!!}^2\frac {1-(-1)^n}2$$

特别定义 $(-1)!!=1$。

> 求 $d_n=[(x+\sqrt{x^2+1})^m]^{(n)}(0)$。

$$y'=m(x+\sqrt{x^2+1})^{m}\frac{1}{\sqrt{x^2+1}}$$

$$y'\sqrt{x^2+1}=my$$

很可惜，$\sqrt{x^2+1}$ 的导数在 $0$ 点也没有特殊性质。

二阶导：

$$y''\sqrt{x^2+1}+y'\frac{x}{\sqrt{x^2+1}}=my'$$

$$my'\sqrt{x^2+1}=m^2y=y''(x^2+1)+y'x$$

求 $n(n\ge 2)$ 阶导：

$$m^2y^{(n)}=y^{(n+2)}(x^2+1)+nxy^{(n+1)}+n(n-1)y^{(n)}+xy^{(n+1)}+ny^{(n)}$$

$$(m^2-n^2)d_{n}=d_{n+2}$$

$d_1=m,d_2=m^2,$ 

当 $n\ge 3$ 且为奇数：

$$d_n=m\prod_{i=1}^{[\frac {n}2]}[m^2-(n-2i)^2]$$

当 $n\ge 3$ 且为偶数：

$$d_n=\prod_{i=1}^{[\frac {n}2]}[m^2-(n-2i)^2]$$

用 [微分计算器](https://www.derivative-calculator.net/) 验算过了，应该没有问题。


------------

> 证明：

> $$(x^{n-1}e^{\frac 1x})^{(n)}=\frac{(-1)^n}{x^{n+1}}e^{\frac 1x}$$

好像不能直接 Leibniz 公式。考虑数学归纳法。我们现在要证明 $n$ 成立，已知的是：

$$(x^{n-2}e^{\frac 1x})^{(n-1)}=\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}$$

不好做。我们对原式下手。

$$(x^{n-1}e^{\frac 1x})^{(n)}=[(n-1)x^{n-2}e^{\frac 1x}-x^{n-3}e^{\frac 1x}]^{(n-1)}$$

$$=(n-1)\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}-(x^{n-3}e^{\frac 1x})^{(n-1)}$$

发现还可以代入一层。（可能得用第二数学归纳更好说明一些。）

$$=(n-1)\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}-\{(x^{n-3}e^{\frac 1x})^{(n-2)}\}'$$

$$=(n-1)\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}-[\frac{(-1)^n}{x^{n-1}}e^{\frac 1x}]'$$

$$=(n-1)\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}+\frac{(-1)^n}{x^n}[\frac 1xe^{\frac 1x}+e^{\frac 1x}(n-1)]$$

$$=(n-1)\frac{(-1)^{n-1}}{x^n}e^{\frac 1x}+\frac{(-1)^n}{x^n}[\frac 1xe^{\frac 1x}+e^{\frac 1x}(n-1)]$$

$$=\frac{(-1)^n}{x^{n+1}}e^{\frac 1x}$$

------------

#### 关于逼近多项式。

对于固定的阶数，如果函数存在逼近多项式，那一定是唯一的。证明作差就可以了。

但是如果一个函数存在 $n$ 阶逼近多项式，那么一定 $n$ 阶可导吗？错的。下面是一个反例（感谢学长！）

考虑狄利克雷函数。

$$f(x)=x^2+D(x)x^3$$

显然函数只在 $x=0$ 处可导。所以二阶导数直接没有定义。

但是函数存在二阶逼近多项式 $y=x^2$，因为

$$\lim\limits_{x\to 0}\frac{f(x)-x^2}{x^2}=\lim\limits_{x\to 0}D(x)x=0$$

（差是二阶无穷小量）

------------

> 设 $f(x)$ 在 $[a,b]$ 上二阶可导，$f(a)=f(b)=0$，证明：

> $$\max_{a\le x\le b}|f(x)|\le\frac 18(b-a)^2\max_{a\le x\le b}|f''(x)|$$

对于任意一个 $x_0\in[a,b]$ ，套路化地泰勒展开：

$$0=f(x_0)+f'(x_0)(a-x_0)+\frac 12f''(\xi_1)(a-x_0)^2,\xi_1\in[a,x_0]$$

$$0=f(x_0)+f'(x_0)(b-x_0)+\frac 12f''(\xi_2)(b-x_0)^2,\xi_2\in[x_0,b]$$

$f'(x_0)$ 不好处理了。我们套路化地取绝对值最大处的 $f(c)$：

$$f(c)=-\frac 12f''(\xi_1)(a-c)^2,\xi_1\in[a,c]$$

$$f(c)=-\frac 12f''(\xi_2)(b-c)^2,\xi_2\in[c,b]$$

两式子相加：

$$2|f(c)|\le\frac 12\max_{a\le x\le b}|f''(x)||2c^2-(2a+2b)c+a^2+b^2|$$

使用二次函数的性质，把 $c=a,c=b$ 代入即可得证。

------------


> ![](https://cdn.luogu.com.cn/upload/image_hosting/1i2l2r5b.png)

经典泰勒展开：

$$f(0)=f(c)+f'(c)(0-c)+\frac12f''(\xi_1)c^2$$

$$f(a)=f(c)+f'(c)(a-c)+\frac12f''(\xi_2)(a-c)^2$$

$$f(c)=M,f'(c)=0$$

$$a=2M+\frac12f''(\xi_1)c^2+\frac12f''(\xi_2)(a-c)^2$$

$$a>0,f(0)+f(a)>0,\therefore M\geq \max(f(0),f(a))>0$$

$$2M=a-\frac12f''(\xi_1)c^2-\frac12f''(\xi_2)(a-c)^2$$

（这一步需要观察，发现答案比 a/2 大！）我们需要二阶导是负的。

而因为原函数有最大值，所以二阶导不可能是正的！（这个用极值点存在的定理可以直接证明）。

$$2M\geq a+\frac1{2a}[c^2+(a-c)^2]\geq \frac 54a$$

证毕。


------------

> $f:[0,h]\to\mathbb{R}$ 上有一阶连续导数，在 $(0,h)$ 上二阶可导，如果 $f(0)=0$，证明：

> 存在 $\xi\in(0,h)$ 使得 

> $$\frac{f(h)-hf'(h)}{h^2} = \frac{\xi f'(\xi) - f(\xi) - \xi^2f''(\xi)}{\xi^2}$$

肯定要构造函数，但是怎么凑成中值定理呢。左边是 $\frac{f(x)}x$ 的导数，但是肯定不能这么搞。我们得想其他方法。但是右边可以直接凑出：

$$\frac{f(x)-xf'(x)}{x}$$

对，右边就是这个的导数。我们定义这个是 $g(x)$，然后再令 $g(0)=0$，这样搞出了一个连续函数。因为对左极限做洛必达：

$$\lim_{x\to0^+}g(x)=-xf''(x)=0$$

那么根据拉格朗日中值定理就有

$$\frac{f(h)-hf'(h)}{h} = \frac{\xi f'(\xi) - f(\xi) - \xi^2f''(\xi)}{\xi^2}$$


------------


------------

> 若 $f(x)$ 在 $[a,b]$ 上连续 $(a>0)$，在 $(a,b)$ 上可导，$f(a)\ne f(b)$，求证：$\exists \xi,\eta\in(a,b),f'(\xi)=\frac{a+b}{2\eta}f'(\eta)$。


盲猜用两次中值定理，然后发现一次柯西一次拉格朗日：

$$\frac{f'(\eta)}{2\eta}=\frac{f(b)-f(a)}{b^2-a^2}$$

$$\frac{f'(\eta)(b+a)}{2\eta}=\frac{f(b)-f(a)}{b-a}$$

$${f'(\xi)}=\frac{f(b)-f(a)}{b-a}=\frac{f'(\eta)(b+a)}{2\eta}$$

------------

> 设 $f(x)$ 在 $[a,b]$ 上二阶可导，$f(\frac {a+b}2)=0$，记 $M=\sup\limits_{x\in[a,b]}|f''(x)|$，证明：

> $$\int_a^bf(x)\mathrm dx\le\frac{M(b-a)^3}{24}$$

先常规地在 $\frac{a+b}2$ 处泰勒展开：

$$f(x)=f'(\frac{a+b}{2})(x-\frac{a+b}2)+\frac 12f''(\xi)(x-\frac{a+b}2)^2\le f'(\frac{a+b}{2})(x-\frac{a+b}2)+\frac M2(x-\frac{a+b}2)^2$$

（因为平方项大于等于 $0$，可以直接代掉。）

积分一下一次项因为对称性变成了 $0$：

$$\int_a^bf(x)\le \frac M2\int_a^b(x-\frac{a+b}2)^2\mathrm dx=\frac M6[(x-\frac{a+b}2)^3\Big|_a^b]=\frac{M(b-a)^3}{24}$$


------------

> 设 $f(x)$ 在 $[0,a]$ 上二阶可导且 $f''(x)\ge 0$，证明：

> $$\int_0^af(x)\mathrm dx\ge af(\frac a2)$$

二阶导？其实很显然，但是不好叙述。所以考虑泰勒展开！

$$f(x)\ge f(\frac a2)+f'(\frac a2)(x-\frac a2)$$

这样两边积分一下就好了。简单明了！

同理可以解决下面一个类似的问题：

> 设 $f(x)$ 在 $[0,1]$ 上二阶可导且 $f''(x)\le 0$，证明：

> $$\int_0^1f(x^2)\mathrm dx\le f(\frac 13)$$

 
------------

> 设 $a_1<a_2<...<a_n$ 为 $n$ 个不同的实数，$f(x)$ 在 $[a_1,a_n]$ 上有 $n$ 阶导数，且 $f(a_1)=f(a_2)=...=f(a_n)=0$，求证：对于 $\forall c\in[a_1,a_n],\exists \xi\in(a_1,a_n)$ 使得

> $$f(c)=\frac{(c-a_1)(c-a_2)\cdots(c-a_n)}{n!}f^{(n)}(\xi)$$

又是一个神题...零点式的神仙应用！

对于一个要证明的 $c$，我们只需要证明存在一个常数 $k$，使得：

$$f(c)-(c-a_1)(c-a_2)\cdots(c-a_n)k=0$$

这其实是显然的。我们接下来需要证明这个常数可以表示为 $\frac{f^{(n)}(\xi)}{n!}$ 的形式。

构造函数 $g(c)=f(c)-(c-a_1)(c-a_2)\cdots(c-a_n)k$，则其有 $n+1$ 个零点，包括 $c$。所以用 $n$ 次 Rolle 定理，可以得到一个 $n$ 阶导为 $0$ 的点。不妨设为 $\xi$ ，那么：

$$g^{(n)}(\xi)=f^{(n)}(\xi)-kn!=0$$

得证！

------------


定积分的一大堆命题复习。

> **黎曼可积的第三充要条件**：
> 对于任意 $\varepsilon>0,\eta>0,\exists$ 划分 $P$ 使得振幅 $\ge\eta$ 的子区间长度之和 $<\varepsilon$。

证明不是很难，可以和第二充要条件互推。我们看一个应用，复合函数的可积性：

> 设 $f(x)$ 在 $[a,b]$ 上可积，$A\le f(x)\le B$，$g(u)$ 在 $[A,B]$ 上连续，证明复合函数 $g(f(x))$ 在 $[a,b]$ 上可积。

根据介值定理首先设 $g(u)$ 属于 $[m,M]$。

那我们取出一个 $f$ 对于 $\varepsilon,\delta$ 的划分，考虑第二充要条件形式：

$$\exists P,\sum_{i=1}^p\omega_i\Delta x_i<\varepsilon$$

不是很好找划分证明。但是 $g(u)$ 的条件是一致连续，这个条件也不是很好用，因为有些区间长度会 $>\delta$，毕竟还有一层嵌套。

考虑在 $f$ 上用第三充要条件，等价于对于任意 $\varepsilon>0,\eta>0,\exists$ 划分 $P$ 使得振幅 $\ge\eta$ 的子区间长度之和 $<\varepsilon$。那么因为 $g(u)$ 一致连续，对于 $\varepsilon>0,\exists\delta>0 \text{ s.t.} |u_1-u_2|<\delta\Rightarrow|g(u_1)-g(u_2)|<\frac{\varepsilon}{2(b-a)}$

对于 $w_i>\delta$，区间长度和 $<\frac{\varepsilon}{2(M-m)}$。划分 $P$ 在 $g(f(x))$ 上的“振幅贡献”（有点不严谨）：

$$\sum_{i=1}^p\omega_i\Delta x_i<\frac{\varepsilon}{2(M-m)}(M-m)+\frac{\varepsilon}{2(b-a)} (b-a)=\varepsilon$$

证毕。

#### 附：

黎曼可积函数的复合不一定可积。例子？

如果 $f,g$ 调换一下也不一定可积。例子？

>  **命题** ：闭区间连续函数可积。

证明需要用 cantor 定理加强条件，然后就可以取振幅直接做了。

>  **命题** ：闭区间单调函数可积。

这个更简单，把区间长度提出来里面直接全消掉了。

>  **命题** ：设 $f(x)$ 在 $[a,b]$ 定义，若 $f(x)$ 仅有有限个不连续点，则 $f(x)$ 在 $[a,b]$ 上黎曼可积。

>  **命题** ：设 $f(x)$ 在 $[a,b]$ 上可积，$g(x)$ 在 $[a,b]$ 上只有有限个点不等于 $f(x)$，有

> $$\int_a^bf(x)\text{d}x=\int_a^bg(x)\text{d}x$$

证明也是简单的，只需要考虑一个划分里面的每个区间是否有“不等点”落在里面。注意贡献要乘上 $2$，因为 Darboux 和每个区间都是闭的。

>  **命题**： 如果定义在区间 $[a,b]$ 有界函数 $f(x)$ 的不连续点有唯一的聚点，那么 $f$ 黎曼可积。

这是习题。证明讨论两种情况，一种是聚点端点，一种是端点。然后根据极限的定义，用一个区间“包住”聚点再构建划分，利用“有连续个不连续点黎曼可积的命题”就证好了。

> **命题**：设 $f$ 在 $[a,b]$ 上极限处处存在为 $0$，证明 $f$ 可积且

> $$\int_a^b f(x)\text{d}x = 0$$

这题其实是黎曼函数可积性证明的一个推广，和测度有关，不往下深究了，习题课讲过，以后大概率要学...

首先有引理：$\forall \varepsilon > 0, |f(x)| > \varepsilon$ 只有有限个 $x$。否则根据聚点原理，会产生聚点，在聚点处 $f$ 的极限不存在或不会为 $0$。设这个点集为 $S$。

那么用套路方法，使用第二充要条件构造划分。设原函数上下确界差为 $M$。

对于任意 $\varepsilon>0$，显然最多只有 $2|S|$ 个区间振幅取到 $M$。

设 $ |f(x)| > \frac{\varepsilon}{2(b-a)}$ 有 $|S|$ 个。取一满足 $\Delta x_i$ 均 $<\frac{\varepsilon}{4|S|M}$ 的分划：

$$\sum_{i=1}^p\Delta x_i\omega_i<2|S|M(\sup \Delta x_i)+(b-a)\frac{\varepsilon}{2(b-a)}=\varepsilon$$

得证。

> 线性性，可乘性，保序性，绝对可积性。

第一个证明显然，取一个划分直接拆。

第二个需要用经典的拆添项构造，和极限的乘积证法是一样的。

第三个证明利用线性性，等价于证明非负可积函数的积分大于等于 $0$。

第四个证明要分为两步，第一步是证明绝对可积，第二部证明不等式。绝对可积使用绝对值不等式 $||a|-|b||\le|a-b|$，然后用第二充要条件。不等式部分先用最简单的绝对值不等式，然后用保序性积一下就好了。

> **积分第一中值定理**： $f(x),g(x)$ 在 $[a,b]$ 上定义且可积，如果 $g$ 不变号，$m\le f(x)\le M$，则 $\exists \eta\in[m,M],\text{s.t.}$

> $$\int_a^b f(x)g(x)\text{d}x = \eta\int_a^b g(x)\text{d}x$$

证明就直接用保序性就好了。常用的是 $g(x)=x$ 的情况。

> **赫尔德不等式的积分形式：** 

假定 $p,q>1,\frac{1}{p}+\frac{1}{q}=1$,函数 $f(x),g(x)$ 在 $[a,b]$ 上黎曼可积，那么

$$|\int_a^bf(x)g(x) \mathrm d x| \le (\int_a^b |f(x)|^p \mathrm d x)^{\frac{1}{p}}\cdot(\int_a^b |g(x)|^q \mathrm d x)^{\frac{1}{q}} $$

证明可以看书。大题是构造两个函数然后再用赫尔德不等式，然后对不等式两边积分。

还有一个 Young 不等式，习题里证明过，这里就不写了。


------------

> 设 $f$ 在 $[a,b]$ 上连续且单调递增，证明：

> $$\int_a^b xf(x)\mathrm dx\ge\frac{a+b}2\int_a^b f(x)\mathrm dx$$

移项：

$$\Leftrightarrow\int_a^b (x-\frac{a+b}2)f(x)\mathrm dx\ge 0$$

因为函数单调，所以发现其实挺显然的，我们用换元积分法就可以轻松证明。

$$\Leftrightarrow\int_a^{\frac{a+b}2} (x-\frac{a+b}2)f(x)\mathrm dx+\int_{\frac{a+b}2}^b (x-\frac{a+b}2)f(x)\mathrm dx\ge 0$$

设 $t=a+b-x,x=a+b-t$：

$$\Leftrightarrow\int_{\frac{a+b}2}^b (\frac{a+b}2-t)f(a+b-t)\mathrm dt+\int_{\frac{a+b}2}^b (x-\frac{a+b}2)f(x)\mathrm dx\ge 0$$

$$\Leftrightarrow\int_{\frac{a+b}2}^b (x-\frac{a+b}2)[f(x)-f(a+b-x)]\mathrm dx\ge 0$$

根据单调性，显然有 $f(x)-f(a+b-x)>0$，得证。


------------

> 设 $f(x)$ 在 $[a,b]$ 上连续可导，且 $f(a)=0$，证明：

> $$\int_a^b|f(x)f'(x)|\mathrm dx\le\frac{b-a}{2}\int_a^b[f'(x)]^2\mathrm dx$$

神题，我不会，太菜了。

首先可以想到对左边凑微分，但是呢，绝对值不能扔进去，因为不知道绝对值函数可不可导。所以我们构造一个可导的函数进行放缩。

设 $g(x)=\int_a^b|f'(x)|\mathrm dx$，显然连续可导，且 $g(x)\ge|\int_a^bf'(x)\mathrm dx|= |f(x)|$，$g'(x)=|f'(x)|$。

$$\int_a^b|f(x)f'(x)|\mathrm dx\le\int_a^bg(x)g'(x)\mathrm dx=\int_a^bg(x)\mathrm d[g(x)]=\frac 12[g(b)]^2$$

$$=\frac 12(\int_a^b|f'(x)|\mathrm dx)^2$$

使用柯西不等式的积分形式（赫尔德不等式积分形式的推论），把其中一个函数当成 $1$ 就可以得到：

$$\le\frac {b-a}2\int_a^b[f'(x)]^2\mathrm dx$$


------------

> 设 $f$ 在 $[0,2]$ 上二阶可导，且 $f(1)>f(0),f(1)>\int_1^2f(x)\mathrm dx$，证明：$\exists \eta\in(0,2),f''(\eta)<0$。

显然使用积分第一中值定理：

$$f(1)>\int_1^2f(x)\mathrm dx=f(\xi),\xi\in[1,2].$$

那么命题就是说，这个函数不可能一直是下凸。考虑反证法。如果 $f''(\eta)\ge0$ 在 $(0,2)$ 上恒成立，也就是 $f(x)$ 是下凸函数（这是书本定理里的充要条件）。由拉格朗日中值定理知 $\exists f'(x_0)=\frac{f(1)-f(0)}{1-0}>0,x_0\in(1,0)$。则根据可导函数单调性，$f''(x)$ 恒 $\ge0 \Rightarrow$ $f'(x)$ 不严格单调递增，$\therefore x\in (x_0,2)$ 时 $f'(x)>0,f(x)$ 单调递增。这与 $f(1)>f(\xi)$ 矛盾！

所以证完了。

------------


求

$$\lim_{n\to\infty}(\frac{2^\frac{1}{n}}{n+1}+\frac{2^\frac{2}{n}}{n+\frac{1}{2}}+\dots+\frac{2^\frac{n}{n}}{n+\frac{1}{n}})$$

考虑凑积分，但是凑不了。但是

$$\lim_{n\to\infty}(\frac{2^\frac{1}{n}}{n}+\frac{2^\frac{2}{n}}{n}+\dots+\frac{2^\frac{n}{n}}{n})$$

显然可以凑积分。极限是

$$\int_0^12^xdx=\frac 1{\ln 2}$$

我们可以惊奇的发现，这两个极限是相等的，因为：

$$\left|\lim_{n\to\infty}(\frac{2^\frac{1}{n}}{n+1}+\frac{2^\frac{2}{n}}{n+\frac{1}{2}}+\dots+\frac{2^\frac{n}{n}}{n+\frac{1}{n}})-\lim_{n\to\infty}(\frac{2^\frac{1}{n}}{n}+\frac{2^\frac{2}{n}}{n}+\dots+\frac{2^\frac{n}{n}}{n})\right|$$

$$=\lim_{n\to\infty}\left|\frac{2^\frac{1}{n}}{n+1}+\frac{2^\frac{2}{n}}{n+\frac{1}{2}}+\dots+\frac{2^\frac{n}{n}}{n+\frac{1}{n}} - \frac{2^\frac{1}{n}}{n}+\frac{2^\frac{2}{n}}{n}+\dots+\frac{2^\frac{n}{n}}{n}\right|$$

$$\le \lim_{n\to\infty} \left|-\frac{2^{\frac{1}{n}}}{n(n+1)} - \frac{2^{\frac{2}{n}}}{2n(n+\frac{1}{2})}-\dots - \frac{2^\frac{n}{n}}{nn(n+\frac{1}{n})}\right| = 0$$

$$\le 2\lim_{n\to\infty} \left|-\frac{1}{n(n+1)} - \frac{1}{2n(n+\frac{1}{2})}-\dots - \frac{1}{nn(n+\frac{1}{n})}\right| = 0$$

$$\le 2\lim_{n\to\infty} \left|\frac{1}{n}\right| = 0$$


------------

> 设 $f(x)$ 在 $[a,b]$ 上连续可导，证明：

> $$\max_{x\in[a,b]}|f(x)|\le|\frac 1{b-a}\int_a^bf(x)\mathrm dx|+\int_a^b|f'(x)|\mathrm dx$$

当时自己做出来了，复习的时候以为这题不可做...

有一个显然的中值定理形式。设：

$$f(c)=\frac 1{b-a}\int_a^bf(x)\mathrm dx$$

$$f(d)=\max_{x\in[a,b]}|f(x)|$$

那么原式：

$$=|f(c)+f(d)-f(c)|\le|\frac 1{b-a}\int_a^bf(x)\mathrm dx|+|\int_c^df'(x)\mathrm dx|$$

$$\le |\frac 1{b-a}\int_a^bf(x)\mathrm dx|+\int_c^d|f'(x)\mathrm |dx$$

$$\le |\frac 1{b-a}\int_a^bf(x)\mathrm dx|+\int_a^b|f'(x)\mathrm |dx$$

------------

> 设 $f(x)$ 是 $[0,1]$ 上的连续函数，证明：

> $$1.\lim_{n\to\infty}\int_0^1x^nf(x)\mathrm dx=0$$

> $$2.\lim_{n\to\infty}n\int_0^1x^nf(x)\mathrm dx=f(1)$$

好像可以分部积分？错了，**这要求 $f(x)$ 可导**，很可惜题目条件比这个强。

第一问很简单套路，分成两段，有一段函数值趋向 $0$，有一段区间长度趋向 $0$。设 $|f(x)|$ 的上界是 $M$。

$M=0$ 显然成立，不作考虑。

$$\int_0^1x^nf(x)\mathrm dx=\int_0^yx^nf(x)\mathrm dx+\int_y^1x^nf(x)\mathrm dx\leq My^n\cdot 1+(1-y)M$$

对于 $\varepsilon>0$，设 $y=\max(0,1-\frac{\varepsilon}{2M})$，那么根据极限的定义，$\exists N>0,$ 有 $n>N$ 时 $y^n<\frac {\varepsilon}{2M}$。那么原式就在 $n>N$ 时 $<\varepsilon$。得证。

第二问就有点不直观了。我们考虑先凑微分：

$$=\lim_{n\to\infty}\frac{n}{n+1}\int_0^1f(x)\mathrm dx^{n+1}$$

$$=\lim_{n\to\infty}\frac{n}{n+1}\int_0^1f(\sqrt[n+1]x)\mathrm dx$$

$$=\lim_{n\to\infty}\int_0^1f(\sqrt[n+1]x)\mathrm dx$$

那么等价于证明

$$\lim_{n\to\infty}\int_0^1[f(\sqrt[n+1]x)-f(1)]\mathrm dx=0$$

那么故技重施，因为当 $x\ne 0$，

$$\lim\limits_{n\to\infty}f(\sqrt[n+1]x)= f(\lim\limits_{n\to\infty}\sqrt[n+1]x)=f(1)$$

分段就可以了！设 $|f(\sqrt[n+1]x)-f(1)|$ 上界为 $M$。

$M=0$ 原式显然成立，不作考虑。

$$\int_0^1[f(\sqrt[n+1]x)-f(1)]\mathrm dx=\int_0^y[f(\sqrt[n+1]x)-f(1)]\mathrm dx+\int_y^1[f(\sqrt[n+1]x)-f(1)]\mathrm dx$$

$$\le My+\sup\limits_{x\in[y,1]}\{f(\sqrt[n+1]x)-f(1)\}\cdot1$$

对于 $\varepsilon>0$，设 $y=\min(1,\frac{\varepsilon}{2M})$，那么根据极限的定义，$\exists N>0,$ 有 $n>N$ 时 $f(\sqrt[n+1]x)-f(1)<\frac {\varepsilon}{2}$。那么原式就在 $n>N$ 时 $<\varepsilon$。得证。


------------

> 设 $f$ 连续且当 $x\ge a$ 时 $f(x)\le \int_a^xf(t)\mathrm dx$。证明：当 $x\ge a$ 时，有 $f(x)\le 0$。

这其实是批了一层皮的构造函数题，根据 N-L 公式， $g(x)=\int_a^xf(t)\mathrm dx$ 在 $[a,+\infty)$ 上必定可导，且导数为 $f(x)$。那么构造函数 $h(x)=e^{-x}\int_a^xf(t)\mathrm dx,h'(x)=e^{-x}[f(x)-\int_a^xf(t)\mathrm dx]\le 0$。$\therefore h(x)$ 在 $[a,+\infty)$ 上单调递减（不一定严格）。$h(a)=0$，所以 $f(x)\le\int_a^xf(t)\mathrm dx\le 0$。



------------

水群看到一个好题：

> 设 $f(x)$ 在 $[0,1]$ 上连续且可导，当 $x\in[0,1]$ 时，$\int_x^1f(x)\mathrm dx\ge \frac{1-x^3}{2}$，证明：

> $$\int_0^1[f(x)]^2\mathrm dx\ge\frac 5{12}$$

我的做法：在群友的提示下想到凑积分+换元。

$$\int_x^1f(x)\mathrm dx\ge \int_x^1\frac{3x^2}{2}$$

设 $g(x)=f(x)-\frac 32x^2$，那么就有 $\int_x^1g(x)\mathrm dx\ge0$。

原式：

$$=\frac 9{20}+\int_0^1[g(x)]^2\mathrm dx+\int_0^13x^2g(x)\mathrm dx$$

只要证明最后一个式子大于等于 $0$ 即可。想到中值定理！由第二中值定理的推论，$3x^2$ 单调增且大于等于 $0$，那么：

$$\int_0^13x^2g(x)\mathrm dx=3\int_\xi^1g(x)\mathrm dx\ge 0$$

证完了！得到了一个更强的界。

学长的做法：构造式子。

设 $\int_1^xf(t)\mathrm dt=F(x)$，则有 $-F(x)>\frac{1-x^3}{2}$。

$$\int_0^1[f(x)-x]^2\mathrm dx\ge\int_0^1[f(x)-\frac{3x^2}{2}]^2\mathrm dx\ge 0$$

左边拆掉用分部积分：

$$\int_0^1[f(x)]^2\mathrm dx\ge 2\int_0^1x\mathrm d(F(x))-\frac 13=2\int_0^1-F(x)\mathrm dx-\frac 13$$

$$\ge 2\int_0^1\frac{1-x^3}{2}\mathrm dx-\frac 13=\frac 5{12}$$


------------

> 设 $f(x)=\sin\frac 1x$，重定义 $f(x)=0$，设 $F(x)=\int_0^xf(t)\mathrm dt$，试求 $F'(0)$。

$0$ 是不连续点，走定义。

$$F'(0)=\lim_{x\to 0}\frac{F(x)}{x}=\lim_{x\to 0}\frac{\int_0^x\sin\frac 1t\mathrm dt}{x}$$

洛出来发现不存在！事实上也没有问题，因为洛不出来但是极限可能存在。比如课本上的例子：

$$\lim_{x\to+\infty}\frac{x+\cos x}{x}$$

所以我们需要换方法。考虑分部积分。

$$\lim_{x\to 0}\frac{\int_0^x\sin\frac 1t\mathrm dt}{x}$$

$$=\lim_{x\to 0}\frac{\int_0^xt^2\mathrm d(\cos\frac 1t)}{x}$$

$$=\lim_{x\to 0}\frac{t^2\cos\frac 1t\Big|_0^x-\int_0^x\cos\frac {2t}t\mathrm dt}{x}$$

$$=\lim_{x\to 0}\frac{x^2\cos\frac 1x-\int_0^x2t\cos\frac 1t\mathrm dt}{x}$$

前面肯定是无穷小量。

$$=\lim_{x\to 0}\frac{-\int_0^x2t\cos\frac {1}t\mathrm dt}{x}$$

然后可以使用中值定理了！考虑第二中值定理的推论，$2t$ 单调递增：

$$=\lim_{x\to 0}\frac{-2x\int_\xi^x\cos\frac {1}t\mathrm dt}{x}$$

$$=-2\lim_{x\to 0}\int_\xi^x\cos\frac {1}t\mathrm dt=0$$