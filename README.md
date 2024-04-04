# PortfolioManager
Methodology

Portfolio management is an important financial practice aimed at achieving optimal returns while mitigating the risks associated with investing in various financial assets. The challenge is to make data-driven  asset allocation and rebalancing decisions under dynamic market conditions. Traditional portfolio management approaches are often based on static models and may not be fully adaptable to changing market dynamics.

 The purpose of this project is to harness the power of deep learning techniques, particularly the Ensemble of Identical Independent Evaluators(EIIE) which uses CNN as a base model to assign correct weights to each asset so that the model can suggest which assets to keep and which to sell from the portfolio. The following are the procedural tasks we shall be implementing to achieve this model:



●	Data Collection and Processing:

We shall retrieve our data from Real-time stock and asset market API sources for training the model. These will include Polygon, Marketwatch and datarade. However, we will not train the model the conventional way. We shall train the EIIE using the Online Stochastic Batch Learning Scheme (OSBL) which includes both past data, pre-trade training, and online real time training. This time asset data will be trained through EIIE and it will output a portfolio vector memory (PVM) for which we shall use a vector store like ChromaDB. The weights of each asset will be corrected and stored in the database. 

●	Using a RL explicit reward function:

The Model’s predictions of weights are corrected on the basis of a RL reward function taking the explicit average of the periodic logarithmic returns. These are then used to evolve the EIIE model to better classify assets to keep or lose in the financial portfolio. 

●	Applying Reinforcement Learning (RL):

Explore the application of EIIE and the explicit reward network in portfolio rebalancing. This requires using the RL as a framework for  asset allocation decisions that maximize returns while respecting risk constraints, which can result in more optimized portfolio performance.

●	Test Robustness:

Evaluate the robustness of the EIIE model in adapting to changing market conditions with additional test data. This includes evaluating whether these models can effectively manage portfolios in real time as market dynamics evolve.





A.	Algorithmic Mathematical Terms and Definitions:



At the beginning of each period, the AI trading agent determines fund allocation between assets.The portfolio will consist of ‘m’ assets. 

In general the price fluctuation depends on the following factors - 

1)	Opening price asset 
2)	Closing prices of assets (vt)
3)	Lowest prices of Assets (vlo)
4)	Highest prices of Assets (vhi)
5)	Volume


‘t’ represents the time period before the next reallocation of asset value.

vi,t represents the closing price of the ith asset during the ‘t’ time period 

vt,lo represents the lowest prices of the period ‘t’ for the assets

vt,hi represents the highest prices of the period ‘t’ for the assets

The prices of all assets are quoted in a standard currency (cash) which in this experiment is taken as the bitcoin.

The vt, vt,hi, vt, low are always one for the quoted currency which in this case is bitcoin.

i.e, v0,t hi = v0,t lo = v0,t = 1, ∀t (trading periods)

For continuous markets, elements of vt are the opening prices for Period t + 1 as well as the closing prices for Period t.




Price relative vector for t (yt) = vt ⊘ vt−1  = 

 


yt elements are ratios of closing prices and opening prices for individual assets in a period. The price relative vector is important in calculating the change in the total portfolio value after a period. If pt-1 is the portfolio value at the beginning of Period t, ignoring transaction cost,
        pt = pt−1 yt · wt−1,

Wt-1 is the portfolio weight vectors at the beginning of a period t whose ith element, wt−1,i, is the proportion of asset i in the P portfolio after capital reallocation.The elements of wt always sum up to one by definition, i wt,i = 1, ∀t

Rate of return for Period t:
  
Logarithmic rate of return for Period t:

 

Initial portfolio weight vector w0 is assumed to be w0 = (1,0,...0)T

 

Where p0 is the initial investment amount and of pf is the final one. We need to maximize pf to obtain the greatest performance in a time frame through the model.

 

The final portfolio value is:

 


B.	Transaction Cost

Transaction cost accounts for the commission fee while purchasing or selling assets in a realistic market setting. We will derive portfolio weights, rate of return and portfolio values accounting for this.

Let’s imagine, at the beginning of period ‘t’, the portfolio vector is wt-1.
Due to price movements, at the end of the period, the weights shift into w’t

 


where ⊙ is the element-wise multiplication. The portfolio manager at the end of period t is to reallocate portfolio vector from w’t to wt by selling or buying relevant assets. Paying all commission fees, this reallocation action shortens the portfolio value by a factor µt where µt ∈ (0, 1] and this is called the transaction remainder factor.

 
Figure 1: This image depicts the change of transaction remainder factor µt.  Market change (Relative prices) is depicted by the vector yt which drives the portfolio value and weights from pt-1 and wt-1 to p’t and w’t. The asset selling and buying action at time ‘t’ redistributes the fund into wt and the transactions as well as the commission fee deducted from pt-1 reduces it to pt by a factor of µt . Rate of return for period t is calculated with portfolio values as shown earlier but this time including the transaction remainder factor (µt).

Now we have to find µt:

pt = µtp’t

Rate of return and logarithmic rate of return (including µ) :

 
The final Portfolio thus, value becomes:

 

The remaining problem is to determine this transaction remainder factor µt . During the portfolio reallocation from w′ t to wt , some or all amount of asset ‘i’ needs to be sold, if p ′ tw ′ t,i > ptwt,i or w ′ t,i > µtwt,i. The total amount of cash obtained by all selling i:

 

Where 0<=s<1 is the commission rate for selling and (x)+ = ReLu(x)
Is the element-wise linear function where if x>0, (x)+ = x otherwise 0.

This money and the original cash p’tw’t,0 taken , the new reserve (µtp ′ twt,0)
will be used to buy new assets.

where 0 <= cp < 1 is the commission rate for purchasing

After assumption of 




the equation is simplified to
 

The presence of µt inside a linear rectifier means µt is not solvable analytically, but it can only be solved iteratively

Theorem of convergence of the transaction remainder vector helps us get the most accurate transaction remainder factor µt ( Ormos and Urb´an, 2013).

The speed on the convergence depends on the error of the initial guest µ⊙. The smaller |µt − µ⊙| is, the quicker the ut(k) converges to µt . When cp = cs = c, there is a practice (Moody et al., 1998) to approximate µt with
 
In training of the neural networks, ut(k) with a constant k is determined. In the back-testing process a tolerant error δ determines the first k such that   
|µ˜ (k) t − µ˜ (k−1) t |  < δ,

In general, they are portfolio vectors of two recent periods and price relative vector -
µt = µt(wt−1, wt , yt).

We assume constant commission rate for selling and purchasing for all non-cash assets to be cs = cp = 0.25 %.
In the end the algorithm agent generates a time-sequence of portfolio vectors which are {w1, w2, · · · , wt } in order to maximize the cumulative capital (pf).


C.	Data Preparation and Price Tensor:

We will go over 11 currencies as mentioned above and retrieve the market data consisting of the low, high and closing prices. We also will describe how we take and format the data from the api into tensor inputs suitable together with normalization and a filling the missing data if any before passing it into the EIIE neural network. 

(i) Price Tensors as Inputs:

The purpose is to develop a portfolio vector as an input to a neural network that will be used to manage historical price data. Xt refers to a three-dimensional tensor that has dimension of f, n, and m representing features, number of input periods, and chosen non-cash items respectively. Therefore, in this scenario, n = 50 as it refers to one day and one hour, which emphasizes new price correlation with the present moment. The unchanged price at the conclusion of this particular time – t is regarded as the final cost of all assets. Nevertheless, absolute prices are normalized by means of the last closest price as market dynamics impact returns. 


 

 


Xt is the set which contains normalized pricing matrices of Vt, V(hi)t, and V(lo)t computed using division operation on each element. Consequently, it becomes a portfolio vector, denoted by w^{t} resulting from the impacts of policies (π), X^{t}, and earlier model-derived numbers. Yt+1 measures the following logarithmic rate of return rt+1 at the end of period t+1. This rate functions as an instantaneous reward that assesses the efficiency of the action produced by the environment determined by Xt under a reinforcement learning based framework.


(ii) Reinforcement Learning Implementation:

The deterministic policy gradient algorithm employed by the RL is used in dealing with the sophisticated issues attached to algorithmic portfolio management. To enable a software portfolio manager (the agent) to make his/her way through a financial market to carry out trading operations is the main goal. Here, consisting of all existing resources and varying demands of investors in the market, it is hard to obtain full information. The approach has been inspired by the philosophies of speculated traders that all vital data has already been factored into the available security prices. Sub-sampling schemes such as asset preselection, periodic feature extraction and history cut off help in addressing this challenge where we are dealing with enormous amounts of historical order data. These schemes result in the generation of the price tensor Xt – 1 condensed version of the market environment.

(iii) Reward Function:

By the end of tf + 1 period, the agent must achieve a maximum portfolio value, pf. Maximization of the average logarithmic accumulated return, Represented in Equation given below , is the stated objective. 

 
In this case, at = wt for the model.
Importantly, this reward function is equitable regardless of different run lengths for the trading policy training using a mini-batch approach. The unique aspect of this reward scheme is that it considers each episode-related reward important for a terminal return. Combining it with the zero-market-impact assumption allows one to speed-up the training procedure while minimizing the problem of local optimum. Using a denominator, tf, in the reward function facilitates training of the policy in mini-batches and allows for flexibility regarding the time horizon applied.
(iv)Deterministic Policy Gradient:
 
By applying the deterministic policy gradient ascent algorithm, one can get the best for the agent. The policy is denoted by a parameter set, θ and it maps deterministically from state space to action space. These parameters continuously adjusted along the length of each of the gradients are determined by learning rate λ. 
 
In order to provide speedy training and work with the flexible structure of modern trading, the updating of parameters takes place not in full data on the training market but in mini-batches. Besides enabling optimal fine tuning of policy parameters this also enables learning online which is really important in an environment where new market history keeps affecting the agent’s decisions.

 



(v) Network Topologies

In the model,  networks take the price tensor Xt as input and produce the portfolio vector wt as output. Both figures illustrate a hypothetical example of the output portfolio vector, while the dimensions of the price tensor, and consequently the number of assets, reflect the actual values used in the experiments. The final hidden layers represent the voting scores for all non-cash assets. The softmax results of these scores, along with a cash bias, determine the corresponding actual portfolio weights. 

To incorporate transaction costs into the neural network's consideration, the portfolio vector from the preceding period, wt-1, is introduced to the networks just before the voting layer.

All the networks flow independently for the different ‘m’ assets while the parameters (weights) get shared. These independent streams separately analyze individual assets and only interconnect at the softmax function to output weights which sum up to 1. This as a whole forms the Independent and Identical Evaluating Networks referred to as EIIE. The network the EIIE will be based on will be a convolutional network implementation. 

EIIE significantly enhances portfolio management performance. In contrast to the integrated network in the previous version, which takes into account the historical performance of individual assets and may be hesitant to invest in historically unfavorable assets, even if they show promise for the future, an IIE is capable of assessing the potential rise and fall of an asset solely based on more recent events. Importantly, the IIE does not disclose the identity of the assigned asset in its judgment process.

EIIE also has benefits like scalability in the number of assets ‘m’ to manage outputs and sharable parameters across a network independently making it data efficient. It can moreover classify any kind of tradable assets and is fully customizable and we can even substitute CNN for networks like RNNs and LSTMs. 

(vi) Portfolio - Vector Memory

 
Figure 2: CNN implementation of EIIE is shown here.  All local receptive fields in the feature maps share a common characteristic in the EIIE configuration, where the first dimensions of these fields are set to 1. This isolation between rows remains until the application of the softmax activation. In addition to the typical CNN feature of weight-sharing among receptive fields within a feature map, an EIIE configuration also involves parameter sharing between rows. Each row in the entire network corresponds to a specific asset and is responsible for providing a voting score to the softmax, indicating the anticipated growth potential of the asset in the upcoming trading period. The input to the network is a 3 × m × n price tensor, encompassing the highest, closing, and lowest prices of m non-cash assets over the last n periods. The network's outputs are the updated portfolio weights. An extra feature map containing the previous portfolio weights is inserted before the scoring layer, allowing the agent to minimize transaction costs.
