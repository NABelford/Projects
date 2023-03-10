library(tidyverse)

var_importances <- read.csv('feature_importances.csv',header=TRUE)

#reading in main dataset
data <- read.csv('flatfile.csv', header=TRUE)

#creating scores based on thresholds listed in var_importances
data_score <- data %>%
              select(FIPSCode,
                     DeathsPerCapita,
                     medianHousingPrice,
                     medianHouseholdIncome,
                     lifeExpectancy,
                     percentPhysicallyInactive,
                     percentChildrenInPoverty,
                     percentFairPoorHealth,
                     percentNonHispanicWhite,
                     percentUninsured,
                     teenBirthRate) %>%
              filter(!is.na(medianHousingPrice)) %>%
              mutate(medianHousingPrice_score = ifelse(is.na(medianHousingPrice) | medianHousingPrice >= 157481, 0,1),
                     medianHouseholdIncome_score = ifelse(is.na(medianHouseholdIncome) | medianHouseholdIncome >= 51870,0,1),
                     lifeExpectancy_score = ifelse(is.na(lifeExpectancy) | lifeExpectancy >= 76.52,0,1),
                     percentPhysicallyInactive_score = ifelse(is.na(percentPhysicallyInactive) | percentPhysicallyInactive >= 25.2,0,1),
                     percentChildrenInPoverty_score = ifelse(is.na(percentChildrenInPoverty) | percentChildrenInPoverty >= 24.12,1,0),
                     percentFairPoorHealth_score = ifelse(is.na(percentFairPoorHealth) | percentFairPoorHealth >= 19.03,1,0),
                     teenBirthRate_score = ifelse(is.na(teenBirthRate) | teenBirthRate >= 30.86,1,0))%>%
              mutate(totalScore = medianHousingPrice_score +
                                 medianHouseholdIncome_score+
                                 lifeExpectancy_score +
                                 percentPhysicallyInactive_score +
                                 percentChildrenInPoverty_score +
                                 percentFairPoorHealth_score +
                                 teenBirthRate_score
                                 )

#plotting raw total score by DeathsPerCapita
plot(data_score$totalScore, data_score$DeathsPerCapita)

#calculating means when grouped by raw total score
scores_means <-data_score %>% select(totalScore, DeathsPerCapita) %>%
                group_by(totalScore) %>%
                summarize(mean_DPC = mean(DeathsPerCapita))

#checkign for a linear relationship between totalScore and DeathsPerCapita
#model and variable is significant R2=0.1504
mod <- lm(DeathsPerCapita*10000 ~ totalScore,
          data=data_score)
summary(mod)

#binning totalScore into 4 bins instead of 7.
#This gives a better delineation of risk in the groups
data_score$totalScore_binned <- ifelse(data_score$totalScore <= 1, '0-1',
                                    ifelse(data_score$totalScore <=3, '2-3',
                                      ifelse(data_score$totalScore <=5, '4-5','6-7')))

#checking for linear relationship and calculating means of DeathsPerCapita in each group
#model and varibale significant. R2 = 0.1554
mod2 <- lm(DeathsPerCapita*10000 ~ totalScore_binned,
          data=data_score)
summary(mod2)

data_score %>% group_by(totalScore_binned) %>%
  summarize(mean_DPC = mean(DeathsPerCapita),
            median_DPSC = median(DeathsPerCapita),
            sd_DPC = sd(DeathsPerCapita))

#boxplot of DeathsPerCapita grouped by totalScore_binned
boxplot(DeathsPerCapita~totalScore_binned, data=data_score,
        main = 'DeahtsPerCapita Grouped By totalScore_binned')


#ANOVA to assess DeathsPerCapita when grouped by totalScore_binned
anova <- aov(DeathsPerCapita~as.factor(totalScore_binned), data=data_score)
summary(anova)
TukeyHSD(anova)

#checking constant variance assumption
plot(anova,1)

library(car)
leveneTest(DeathsPerCapita~totalScore_binned, data=data_score)

pairwise.t.test(data_score$DeathsPerCapita,
                data_score$totalScore_binned,
                pool.sd = FALSE)

#checking normality assumption
plot(anova,2)

anova_resid <- residuals(object=anova)
shapiro.test(anova_resid)

kruskal.test(DeathsPerCapita ~ totalScore_binned, data=data_score)

