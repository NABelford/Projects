library(stringr)
library(dplyr)
library(rtweet)
library(readxl)


#############################
## Scraping Twitter Data ####
#############################


df <- read_excel('twitter_users.xlsx', sheet=1, col_names = TRUE)

#authenticating twitter session/app
twitter_token <- create_token(
  app = 'MSIS5193_Project',
  consumer_key = 'XXXXXXXXXXXXXXXMaLM2bKC',
  consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXtw4QL7',
  access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXATceLnjO',
  access_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXZ0WNf')

#Removing rows with no twitter handle for the purpose of collecting twitter data
#Twitter data will be added to the original dataset where all movies remain
#movies without twitter data will have NA's in those columns
df_clean <- df %>% filter(twitter_name != 'not_available')

#checking that all twitter_name column values are unique (so no duplicates like NA or 'unavailable')
length(unique(df_clean$twitter_name)) == nrow(df_clean)

#Initializing Dataframe onto which we add each movie's info
#daily_list <- data.frame(NULL)

tweet_list <- data.frame(NULL)

counter = 0

#The data set contains too many twitter handles to pull the entire dataframe at once
#as the limit is 18k tweets per 15 minutes according to the documentation.
#The following will create 5 evenly spaced numbers to break up the data into 4 even parts.
#Each part will take about 5 minutes to pull, then you will need to wait 5-10 minutes
#and then run the next for loop. This should avoid the rate limit.

#Creates five evenly spaced numbers from 1 to the number of rows in df_clean
stops <- round(seq.int(from = 1, to = nrow(df_clean), length.out = 5),0)

data1 <- df_clean[c(1:stops[2]),]

batch_counter = 0

for(user in data1$twitter_name){
  
  #Tracking which user is being pulled
  counter <- counter + 1
  batch_counter <- batch_counter + 1
  
  print(paste0("Starting Overall User #",counter," of ", nrow(df_clean),": ",user))
  print(paste0("Batch 1: ", batch_counter, " of ", nrow(data1)))
  
  #Pull Twitter Timeline
  timeline <- get_timeline(user, n=18000)
  
  #Printing number of tweets pulled
  print(paste0("Tweets Pulled: ",nrow(timeline)))
  
  if(nrow(timeline) > 0){
    
    #Removing unecessary columns and creating day_created from created_at
    #Really it is removing the time, so that only the date remains
    x <- timeline %>% select(screen_name, text, is_quote, is_retweet,
                             retweet_count, quote_count, reply_count, hashtags,
                             lang)
    
    
    #append the results to the big, ongoing dataframe
    tweet_list <- bind_rows(tweet_list, x)
    
  }
}

#==================
#=== STOP =========
#==================
#Wait 5-10 minutes and then proceed with next loop

data2 <- df_clean[c((stops[2]+1):stops[3]),]
batch_counter = 0

for(user in data2$twitter_name){
  
  #Tracking which user is being pulled
  counter <- counter + 1
  batch_counter <- batch_counter + 1
  
  print(paste0("Starting Overall User #",counter," of ", nrow(df_clean),": ",user))
  print(paste0("Batch 2: ", batch_counter, " of ", nrow(data2)))
  
  #Pull Twitter Timeline
  timeline <- get_timeline(user, n=18000)
  
  #Printing number of tweets pulled
  print(paste0("Tweets Pulled: ",nrow(timeline)))
  
  if(nrow(timeline) > 0){
    
    #Removing unecessary columns
    x <- timeline %>% select(screen_name, text, is_quote, is_retweet,
                             retweet_count, quote_count, reply_count, hashtags,
                             lang)
    
    
    #append the results to the big, ongoing dataframe
    tweet_list <- bind_rows(tweet_list, x)
    
  }
}

#==================
#=== STOP =========
#==================
#Wait 5-10 minutes and then proceed with next loop

data3 <- df_clean[c((stops[3]+1):stops[4]),]
batch_counter = 0

for(user in data3$twitter_name){
  
  #Tracking which user is being pulled
  counter <- counter + 1
  batch_counter <- batch_counter + 1
  
  print(paste0("Starting Overall User #",counter," of ", nrow(df_clean),": ",user))
  print(paste0("Batch 3: ", batch_counter, " of ", nrow(data3)))
  
  #Pull Twitter Timeline
  timeline <- get_timeline(user, n=18000)
  
  #Printing number of tweets pulled
  print(paste0("Tweets Pulled: ",nrow(timeline)))
  
  if(nrow(timeline) > 0){
    
    #Removing unecessary columns and creating day_created from created_at
    #Really it is removing the time, so that only the date remains
    x <- timeline %>% select(screen_name, text, is_quote, is_retweet,
                             retweet_count, quote_count, reply_count, hashtags,
                             lang)
    
    
    #append the results to the big, ongoing dataframe
    tweet_list <- bind_rows(tweet_list, x)
    
  }
}
#==================
#=== STOP =========
#==================
#Wait 5-10 minutes and then proceed with next loop

data4 <- df_clean[c((stops[4]+1):stops[5]),]
batch_counter = 0

for(user in data4$twitter_name){
  
  #Tracking which user is being pulled
  counter <- counter + 1
  batch_counter <- batch_counter + 1
  
  print(paste0("Starting Overall User #",counter," of ", nrow(df_clean),": ",user))
  print(paste0("Batch 4: ", batch_counter, " of ", nrow(data4)))
  
  #Pull Twitter Timeline
  timeline <- get_timeline(user, n=18000)
  
  #Printing number of tweets pulled
  print(paste0("Tweets Pulled: ",nrow(timeline)))
  
  if(nrow(timeline) > 0){
    
    #Removing unecessary columns and creating day_created from created_at
    #Really it is removing the time, so that only the date remains
    x <- timeline %>% select(screen_name, text, is_quote, is_retweet,
                             retweet_count, quote_count, reply_count, hashtags,
                             lang)
    
    
    #append the results to the big, ongoing dataframe
    tweet_list <- bind_rows(tweet_list, x)
    
  }
}

#The timeline pulls the hashtags as a list of hashtags within each cell.
#The following turns them into a single string separated by commas

tweet_list$hashtags2 <- sapply(tweet_list$hashtags, 
          function(x) paste(unlist(x), collapse = ','))

tweet_list <- tweet_list %>% select(-hashtags) %>%
                    rename(hashtags = hashtags2)

#writing tweet_list as a comma separated file. COlumns 2 and 8 (text and hashtags)
#are quoted because of the punctuation within each element of the column.

write.csv(tweet_list, 'MovieTweets_Original.csv',
            row.names=FALSE, quote = c(2,8))

#Test call to read in newly created file to check for errors
#test <- read.csv('MovieTweets_Original.csv', header=TRUE)
