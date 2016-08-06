# load in the training data
genderAgeTrain <- read.csv('~/Development/data/talking_data/gender_age_train.csv')
events <- read.csv('~/Development/data/talking_data/events.csv')
app_events <- read.csv('~/Development/data/talking_data/app_events.csv')

# merge events and app data together
eventsWithAppData <- merge(events, app_events, by='event_id')

# compute number of events by device
eventsByDevice <- aggregate(event_id ~ device_id, data = events, FUN = length)

# compute mean installed apps by device
installedByDeviceAndEvent <- aggregate(is_installed ~ event_id + device_id,
                                       data = eventsWithAppData,
                                       FUN = length)
colnames(installedByDeviceAndEvent) <- c('event_id', 'device_id', 'installed_count')

meanInstalledByDevice <- aggregate(installed_count ~ device_id, 
                                   data = installedByDeviceAndEvent,
                                   FUN = mean)
colnames(meanInstalledByDevice) <- c('device_id', 'mean_installed_count')

# compute mean active apps by device
isActiveByDeviceAndEvent <- aggregate(is_active ~ event_id, device_id,
                                      data = eventsWithAppData,
                                      FUN = sum)
meanActiveByDevice <- aggregate(is_active ~ device_id,
                                data = isActiveByDeviceAndEvent,
                                FUN = mean)
colnames(meanActiveByDevice) <- c('device_id', 'mean_active_count')


# build merge into training set
trainData <- merge(genderAgeTrain, eventsByDevice, by = 'device_id', all.x = TRUE)
trainData <- merge(trainData, meanActiveByDevice, by = 'device_id', all.x = TRUE)
trainData <- merge(trainData, meanInstalledByDevice, by = 'device_id', all.x = TRUE)
trainData <- trainData[, c(1,4,5,6,7)]

trainDataImputedWithMeans <- trainData
trainDataImputedWithNegOne <- trainData

trainDataImputedWithMeans[is.na(trainDataImputedWithMeans$event_count), c('event_count')] <- mean(trainDataImputedWithMeans$event_count, na.rm= TRUE)
trainDataImputedWithMeans[is.na(trainDataImputedWithMeans$mean_active_count), c('mean_active_count')] <- mean(trainDataImputedWithMeans$mean_active_count, na.rm= TRUE)
trainDataImputedWithMeans[is.na(trainDataImputedWithMeans$mean_installed_count), c('mean_installed_count')] <- mean(trainDataImputedWithMeans$mean_installed_count, na.rm= TRUE)
trainDataImputedWithNegOne[is.na(trainDataImputedWithNegOne)] <- -1

phones <- read.csv('phone_brand_device_model.csv')

trainDataImputedWithNegOne <- merge(trainDataImputedWithNegOne, phones, by='device_id')
trainDataImputedWithMeans <- merge(trainDataImputedWithMeans, phones, by = 'device_id')


# write.csv(trainDataImputedWithNegOne, file = 'train_data_imputed_neg_ones.csv')
# write.csv(trainDataImputedWithNegOne, file = 'train_data_imputed_neg_ones.csv', row.names = FALSE)
# write.csv(trainDataImputedWithMeans, file = 'train_data_imputed_means.csv', row.names = FALSE)


