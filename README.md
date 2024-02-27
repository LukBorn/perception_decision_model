Decision Model based on Lak et al 2019:
Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon
https://elifesciences.org/articles/49834

WORKS:
decision model (figure 3a in Lak et al 2019)
psychometric(stimulus vs choice average), subset psychometric for each previous stimulus, difference matrix, hard vs easy choice updating % (figure 1d in Lak et al 2019)
psychometric for each previous choice (figure 1c in Lak et al 2019)
exploration of alpha and sigmas effects on updating matrix

SEMI-WORKS:
for some reason the model updating percentages are a lot lower than in the lak et al paper
in the subset psychometric and previous choice psychometric the expected shift is not really visible as much as it should be
alpha_sigma is broken somehow


TODO:
plot reward prediction error in abhängigkeit von evidenz (stimulus)

implement proper reward bias blocks
function for generating blocks 
plot psychometric in a meaningful way to show that the model adapts to bias blocks -> maybe psychometric for each block
psychometric for previous lean/previous

implement a way to model time investment
timeinvestment(value, confidence) = a*value+exp(b*confidence)
look at time investment task design
look at previous attempts to implement time investment
