# Postmortem: Hybrid Deep Learning GARCH for Stablecoin Volatility Forecasting

This document records what went wrong in this project, why, and what the next steps would be. Decisions are evaluated both with information *at the time* and *in hindsight*. As the thesis is an official university assignment with a deadline, new knowledge that I acquired after the submission of this thesis would fundamentally impact a rerun. Thus, the thought process at the time needs to be evaluated given contemporaneous knowledge.

> Companion reading: 📝 [README](README.md) for the project overview · 📓 [Data diagnosis notebook](notebooks/diagnostic_test.ipynb) for statistical evidence · 📄 Chapter 6.4 of the [Thesis paper](papers/thesis.pdf) for formal discussion of limitations

---

## Summary

The thesis/project set out to test whether a neural network conditioned on market microstructure would produce more accurate volatility forecasts compared to the standard GARCH on stablecoin pairs. The 12-day USDC/USDT sample contained no ARCH effects, thus leaving both econometric and hybrid GARCH unidentifiable. 
The root cause was a data problem, not a modeling problem, as the pair was too stable and the window too calm and short. The most important lesson I took from this project is all the mistakes I made in data collection. From testing the data to foolproofing the data collection infrastructure, everything fails if the data is off.

## What was supposed to happen

My hypothesis was that processes and conditions underlying the cryptocurrency exchange market structurally drive volatility. As such, a deep learning model conditioned on 41 microstructure features (from CEX, DEX and on-chain states) should produce *time-varying* GARCH parameters α<sub>t</sub>, β<sub>t</sub> that track the state of the market. For example, periods of low liquidity should result in a higher shock parameter α, as the impact trades have on price movements increases. If market microstructure influences volatility, this hybrid model should beat the standard GARCH(1,1) out of sample.

Implicitly, I assumed that my data would have statistically significant ARCH effects.

## What actually happened (Which problems did I run into)

In rough chronological order:
1. **Data collection ran from a machine inside mainland China.** Since I did my thesis at Peking University, I was also physically in mainland China, where (retail) cryptocurrency trading is banned. Binance Websockets and several APIs I used for data collection are not reachable without a VPN, and the VPN connection dropped repeatedly, especially during nighttime, where I could not manually restart my VPN. This left me with 23 data gaps of varying size (see **Chapter 4.3** of the [thesis](papers/thesis.pdf) for more information).
2. **The gaps eliminated memory-based architectures.** The original plan was to run an LSTM (long-short term memory) deep learning model instead of a feedforward NN to capture the change in states. Due to the scattered data gaps, a memory-based model would be severely limited in clean observations without gaps in the memory. Therefore, the feedforward architecture was used because other architectures could not be. 
3. **The softmax parameterization collapsed to α = 1.** Regardless of the learning rate, epochs, or weight decay, the softmax runs converged to α ≈ 1 and β ≈ 0. More on this in *root cause 3*.
4. **The switch to two independent sigmoids showed a deeper problem.** Switching to sigmoids solved the issue that α converged to 1, but now both α and β converge to near zero. Comparing it to the GARCH benchmark, which collapsed to a near-integrated process, made it clear that the issue lay in the data.
5. **Tests confirm: the underlying data showed no ARCH effects.** As elaborated in more detail in the [data diagnosis notebook](notebooks/diagnostic_test.ipynb), the collected data showed no ARCH effects. As such, any model run on this data would lead to a similar result. 
6. **Permutation features still show theoretically coherent results.** The most salvageable result is that the model appears to have learned the most mechanisms laid out in the thesis/ordered them as expected. 


## Why did it happen (root cause analysis)

1. **Fundamental issue: The chosen stablecoin pair was too non-volatile and the coins too similar**

The reason I chose USDC/USDT is that I did not want to estimate the conditional mean. Since USDC and USDT both track the US dollar, they should, economically speaking, trade in lockstep, which they do not do. From that I inferred that the deviations between the two coins must be attributable to the mechanisms of the underlying market and thus be the easiest way to measure mechanical volatility. In hindsight, I believe that there were two issues: too little trading activity and too well-functioning an arbitrage system. 

I believe the issue was that I was basically comparing Euros printed in Germany to Euros printed in Italy, which begs the question of why anyone would trade this pair to begin with. Both coins serve almost the same function, are pegged to the same asset, and are both backed by large asset pools. Furthermore, at the time of writing the thesis, the v3 pool still had the highest 30D volume of around $520M out of the USDC/USDT pools, but the introduction of v4 already started pulling away some trading activity and volume towards new v4 pools. ARCH effects require informed trading, of which there is none, and inventory pressure for which the volume is too low.

Secondly, I underestimated the speed at which arbitrage on Uniswap occurs and misread mempool congestion. I can only observe ARCH effects if shocks persist, and I thought that due to mempool congestion, these shocks would carry over to the next 5 min interval, basically elongating the time at which shocks skew the exchange rate. It does not. Congestion still plays a role in widening the no-arbitrage band, thereby requiring a larger deviation for arbitrage to be profitable, which is not really a duration effect. Secondly, the speed at which transactions become added to the block is much too short (12 to 24 seconds) for the 5 min interval I sampled.
I did not need slower arbitrage but instead should have looked at a time when arbitrage stops working, but more on that in point 2. 

2. **Fundamental issue: The sample was too calm and too short**

Going back to point 1, what my sample lacks is any periods of peg stress and strong deviations from parity, such as Q1 2023. First and foremost, the period I sampled was extremely calm (even in stablecoin standards), and thus the model did not really have any volatility to learn. Even if the mechanism would not work as well, there just was not enough volatility and trading to train a GARCH. More importantly, though, a period of peg stress would take the mechanism out of commission, thereby potentially producing ARCH effects.

Aside from the data quality, the model would also most likely not have been trained properly with this size of the sample. I would need some 25,000 observations (or about 90 days) minimum for the 2,467 parameters I use in my model.

3. **Incident: Sigmoid vs. softmax**

When choosing the activation function, I liked softmax because it exponentially amplifies the highest score and because I associated sigmoids with binary decisions. This was incorrect because a microstructural signal can and should not fully be attributed either to α or β. What softmax does is create a zero-sum game, which does not fit into a GARCH context, as α and β are economically independent and must not be forced against each other. 

With this setup there are two explanations for α approaching 1 while β goes to 0: Firstly, shocks always "overpower" persistence, which means that since α multiplies the previous period's squared return, likelihood attributes the signal to increase α. β influences through recursion, and thus the next period will again be attributed to α. Secondly, the lack of ARCH effects means there is no volatility clustering. Naturally, all the signal thus gets attributed to α. The issue here is that the model appears to work (i.e., attributes everything to shocks), masking the fact that there are deeper problems (no ARCH effects).

Also, since I am not running an IGARCH, there was no reason to want the sum to equal 1; this was an oversight.

I only call this an incident because it did not affect the outcome of the thesis, and the sigmoid change revealed the more important findings.

---

## What would I change

In hindsight, there are a couple of changes I would make if I were to rerun this thesis sorted by relevance:

1. **Check the data before building anything.** Run an ARCH LM or Ljung-Box test before committing time to building a model that does not work. Realistically, this would not have saved the thesis, but it would have changed what I build for a rerun. For the longest time, I believed that there were insurmountable barriers in data availability, i.e., no available historical DEX pool data, tick data, or mints / burns data. Turns out, it is historically queryable back to May, 2021. The CEX order book is ephemeral.

2. **Make sure the data collection architecture never breaks.** Working with data gaps introduces unnecessary issues, which now that I am not bound by VPN issues can be completely avoided. Alternatively, if I were still in China, I would use a virtual private server (e.g., in Germany) to run data collection and access it through a VPN. I remember thinking about the VPS solution during my thesis too, but I unfortunately decided against it since I thought the VPN would hold.

3. **Change the pair or sample a more turbulent market.** As discussed in *root causes* 1 and 2, I could either change to a different currency pair, such as ETH/USDC, and do the conditional mean estimation or attempt to rerun the USDC/USDT pair with a sample from a more volatile period; there are merits to both if the rerun would not focus on stablecoins. To run the same model on a different asset pair, one would obviously have to estimate the mean through ARMA or related models and then test the benchmark GARCH and hybrid GARCH relatively against each other on MSE, RMSE, or MAE.  

4. **Collect more observations.** Pretty self-explanatory, as explained in *root cause* 2. The only caveat is that DuneAnalytics operates on credits for their queries. Therefore, the queries either need to be trimmed down or run over different months. 

5. **Investigate memory-based architectures.** With implementing points 2 and 4, I should have enough observations with as few gaps as possible to reliably train an LSTM model and compare it to the feedforward model. I do think that a memory-based architecture would have been the better option if available. LSTM is also often covered in architecture in similar academic work (for example, see [García-Medina & Aguayo-Moreno, 2024,](https://link.springer.com/article/10.1007/s10614-023-10373-8) or [Amirshahi & Lahmiri, 2023](https://www.sciencedirect.com/science/article/pii/S266682702300018X)).

## What would I keep

The methodology was sound, and it is worth keeping intact, as the issue lies with the data, not the model itself:

- **Chronological 70/10/20 training split.** The 70/10/20 training split is sound.
- **Scaler fit on the training split only.** No leakage from later periods into feature representation.
- **Using the same Gaussian NLL objective for benchmark and hybrid.** It isolates architecture from the objective. Thus, differences are attributable to architecture alone.
- **The permutation-importance analysis.** It ordered the used signals theoretically coherent. While the ordering is more suggestive for the model learning correctly, it is a very important sanity check and can be used to evaluate if the theoretical mechanisms actually work. 
- **Scaling the hybrid's outputs with the benchmark GARCH's outputs.** It did not do too much here, but I do believe in this idea, and I would reuse it in a rerun. (Obviously, since both models failed, I used benchmarks based on literature as my bounds in the thesis, but I still think the idea is reasonable.)
- **Using the Adam training algorithm.** Here again, it did not do much here, but I believe here it is a well-fitting algorithm, due to its adaptive nature based on the scale and frequency of inputs.

---

## Lessons

1. **Check my data.** This is the easiest thing to do to avoid a repeat of what happened here. As described in *What I would change 1.* running an ARCH LM and Ljung-Box is quick and simple and should always be done.

2. **The data collection setup must work continuously without interruptions.** Having gapless data would have allowed me much more freedom in choosing architecture and simplified the cleaning substantially. Also, it would have removed the need to reset to unconditional variance every time I had a gap. If that means setting up a VPS, it is worth it.

3. **Know what data is ephemeral and what can be queried retrospectively or reconstructed.** I started my data collection quite late because, for the longest time, I thought I could get historical data for order books and Uniswap, but while writing the thesis, I could not find the workaround I now know. Stream the ephemeral data and spend time finding a way to reconstruct as much as possible.

4. **Never use softmax on signals that can have multiple economic effects.** Specific, but probably useful in the long run. Applies to any attribution mechanism that forces a zero-sum game as well.

5. **Crunch down the economic reasoning; consider the regime you are in.** Choosing stablecoins made a lot of sense to reduce economic volatility; comparing two US dollar-denominated ones maybe not, once you ask why anyone would even trade that pair. It could have worked had I taken a less calm period to sample. Overall, I need to think more about things like regimes and economic reasoning even when working quantitatively.