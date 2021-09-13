source("function/preprocess_class_data.r")
preprocess_class_data(data)
print(paste0("Save the Prepared Data..."))
write.csv(prep,"data/prep.csv")
