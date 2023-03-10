library(tidyverse)

wheel <- c(rep(1,12), rep(3,6), rep(5,4),10,10,20)

outcomes <- c(1,3,5,10,20)
winnings <- c(-2,2,4,-2,-2)

score_matrix <- data.frame(outcomes, winnings)

bankroll_graph <- data.frame(outcome = c(0, rep(0,10000)), winnings = c(0, rep(0,10000)), new_bankroll=c(100, rep(0,10000)))



bet <- 1
score_matrix <- data.frame(outcomes, winnings*bet)

num_sims <- 5000
bets_per_sim <- 100

bg2 <- data.frame(NULL)

for (j in 1:num_sims){
  
  bankroll <- 0
  to_bg2 <- data.frame(sim = rep(j,bets_per_sim),
                       spin = c(1:bets_per_sim),
                       bankroll = c(bankroll, rep(0,bets_per_sim-1)))
  
  for(i in 1:(bets_per_sim-1)){
    num <- sample(wheel, 1, replace=TRUE)
    out <- score_matrix$winnings[score_matrix$outcomes == num]
    bankroll <- bankroll + out
    
    to_bg2$bankroll[i+1] <- bankroll
    
  }
  bg2 <- bind_rows(bg2, to_bg2)
}

bg2 %>% group_by(spin) %>%
  arrange(desc(spin)) %>%
  summarize(avgBankroll = median(bankroll),
            minBankroll = min(bankroll),
            maxBankroll = max(bankroll),
            bankroll90 = quantile(bankroll, 0.9),
            bankroll10 = quantile(bankroll, 0.1)
  ) %>%
  ggplot(aes(x = spin)) +
  geom_line(aes(y = avgBankroll), size =1, color='black') +
  geom_line(aes(y = bankroll10), size =1, color='red') +
  geom_line(aes(y = bankroll90), size =1, color='green') +
  geom_line(aes(y = minBankroll), size =1, color='red3') +
  geom_line(aes(y = maxBankroll), size =1, color='green3') +
  labs(y='Bankroll')

bg2 %>% filter(spin==100) %>% ggplot() + geom_bar(aes(x=bankroll)) 

