# BoostMIS
## Abstract
提出了新的半监督SSL框架  ->  BoostMIS：将自适应未标记和信息主动注释结合
1. 根据当前学习状态自适应地利用未标记数据的聚类假设和一致性正则化。该策略可以根据任务模型预测自适应生成一个热“硬”标签，以便更好地训练任务模型。
2. 对于未选择的低置信度未标记图像，我们引入了一种主动学习（AL）算法，利用虚拟对抗扰动和模型的密度感知熵来寻找信息样本作为注释候选。这些信息丰富的候选者随后将被输入下一个培训周期，以更好地传播SSL标签。值得注意的是，自适应伪标签和信息主动注释形成了一个相互协作的学习闭环，以提高医学图像SSL
## Introduction
1. 数据利用率低：大量未标记的数据被忽略
2. 缺少信息性示例：被忽略的数据可能是有意义的

主动学习AL：选择信息最丰富的样本，以最小的标记成本最大化模型性能（从未标记数据中获取高可信度知识）
1. 医学图像任务模型 Medical Image Task Model（本文中的分类模型）：通过带监督标签的弱增强医学图像训练任务模型
2. 基于一致性的自适应标签传播器 Consistency-Based Adaptive Label Propagator：使用未标记和一致性正则化来传播未标记数据的标签信息。模型的性能在每个阶段都不同，因此根据当前的学习状态定义了自适应阈值用来生成伪标签。一致性正则化使模型对弱增强和强增强数据产生相同预测，增强泛化能力。
3. 对抗不稳定选择器 Adversarial Unstability Selector：通过AL增强SSL。引入虚拟对抗扰动来选择位于聚类边界上的不稳定样本作为注释候选。聚类边界上的样本，SSL模型判别能力很弱，很难区分。通过样本与增加虚拟扰动后的样本之间的不一致性来区分
4. 平衡不确定度选择器 Balance Uncertainty Selector：进一步利用数据。使用SSL中的密度感知熵（density-aware entropy）均匀选择每个预测类中具有高不确定性的样本作为补充集，平衡后续训练（挑选每一类中样本质量较差的）
## Related Work
1. SSL：labeled + unlabeled, eg:熵最小化、 伪标记（自我训练） 、一致性正则化
2. SSL in MIA
3. Semi-supervised Active Learning
## Method
![模型整体框架](./Fig/Framework.png)

#### 1. 术语
标签样本集，带标签的训练集（X,Y），X为样本，Y为标签
无标签训练集U
用（X,Y）和U训练BoostMIS
$$M\left ( \left ( X,Y,U \right );\left ( \ \Theta _{S},\Theta _{A}\right ) \right )=M\left ( \left ( Y,P^{S}\right )|\left ( \ X,U^{S}\right ) \right );\Theta _{S})$$
T是SSL图像训练模型（SSL分类模型），S是通过AL的注释候选
a. 在SSL图像分类中（T）：弱增强X进行训练，并将标签信息通过基于一致性的自适应标签传播器传播到U中，选择后验概率高于自适应阈值的U作为带有伪标签PS的Us，和X一起训练T，并进行一致性正则化
b. 在主动学习AL中：SSL完成后，将剩下的无标记数据Uu添加扰动，输入T，在对抗不稳定选择器中生成对抗样本，然后选择Uu和他们的对抗样本之间KL差异最大的topK样本。
为进一步识别无标记数据，使用T的熵，均匀选择每个类中具有高不确定性的topK样本作为补充集（挑质量差的样本）。最终剩下的候选C（均匀选择）提供给人类专家手工标注

#### 2. CNN，backbone=Resnet50
带弱增广的标记数据（X,Y），比如仅旋转和位移数据增广，训练
$$l^{_{s}}(\Theta _{S})=\frac{1}{N_{t}}\sum_{i=1}^{N_{t}}D_{ce}(y_{i},P_{m}(p_{i}|A_{w}(x_{i}))))$$
Pm：模型的后验概率分布，Dce：两个概率分布之间的交叉熵，pi：预测标签，
Aw：弱增强数据

#### 3. Consistency-based Adaptive Label Propagator-基于一致性的自适应标签传播器
当置信度高于定义阈值时，根据模型预测计算未标记图像的伪标签以及标记图像的伪标签
引入自适应阈值AS：
$$\in t=\left\{\begin{matrix}
\alpha \cdot Min\left \{ 1,\frac{Count_{\in t}}{Count_{\in t-1}} \right \}+ \frac {\beta \cdot N_{A}}{2K}, if t < T_{max}\\ 
\alpha +\beta, otherwise 
\end{matrix}\right.$$
其中，α和β时给定阈值，NA为AL注释候选数量，Tmax是迭代学习的给定值，Nu是未标记数据的数量
$$U={u_{i}|_{i=1}^{N_{u}}}$$
$$Count_{\in t}=\sum _{i=1}^{N_{u}}\mathbb{I}(P_{m}(p_{i}|A_{w}(u_{i}))>\alpha +\beta )$$
Count记录U中高于阈值的的数量。t动态调整，知道迭代步长超过Tmax
然后将弱增强的U输入T。当所选样本的预测置信度高于自适应阈值时，转化为 Us->Ps（伪标签）。然后引入一致性正则化，计算模型对同一图像的强增强的预测。
两者进行预测，计算交叉熵，匹配伪标签
$$l_{u}(\Theta _{s})=\frac{u}{N_{u} ^{s}}\sum_{i=1} ^{N_{u}^{s}}D_{ce}(P_{m}(p_{i}^{s}|A_{w}(u_{i}^{s})),P_{m}(p_{i}^{s}|A_{s}(u_{i}^{s})))$$
弱增强生成标签，强增强用于计算损失
注意：弱增强与强增强
弱：平移、旋转
强：CutOut、CTAugment、RandAugment  对比度、颜色
![强增强](./Fig/img_enhance.png)

#### 4. 对抗不稳定选择器AUS
SSL模型标签传播器可以将标签信息从标记数据传递到伪标记样本
未选择的样本中含有丰富信息，对其进行标注更有价值
在未选择的样Uu中，分为两类：不稳定 & 不确定
使用AUS查找不稳定样本 -> 计算样本prediction和相应的对抗样本prediction之间的不一致性来估计模型对样本prediction的稳定性
具体：未选定的Uu，任务模型最后层的表征ru，最后预测为pu。
将ru和pu输入生成器，得到对抗扰动rp，扰动表征=ru+rp，将扰动表征输入任务模型，得到扰动预测
$$r_{i}^{p}=argmax_{\Delta r_{i}||\Delta r||\leq r}Div(P_{m}(p_{i}^{u}|r_{i}^{u},P_{m}(\bar{p}_{i}^{u}|r_{i}^{u}+\Delta r))$$
tao是扰动步长，Div用来测量两个分布之间的散度，这里用KL散度，而后计算方差
最后，AUS从不稳定样本中选取方差最大的topK样本作为AL注释候选的初始召回集。
这些不稳定的样本通常在聚类群的边缘，这样可以平滑SSL的决策边界，增加预测正确率

#### 5. 平衡不确定度选择器BUS
用来处理剩下的大量不确定样本（较低的预测置信度）
引入BUS，在每个预测类中均匀选择具有高不确定性的未选定样本（用任务模型的熵来估计不确定性），然后均匀选择熵最大的topK样本进行人工标注
熵计算：
$${Ent}'(u_{i}^{u};\Theta _{S})=\sum_{c\in C}P_{m}(p_{i}^{c}|A_{w}(u_{i}^{u}))logP_{m}(p_{i}^{c}|A_{w}(u_{i}^{u}))$$


## Experiment
![实验数据](./Fig//result.png)