# normal 462
'Test_loss': 0.05413860082626343, 'Train_f1_score': 0.8628391591066702,  'Test_f1_score': 0.6404338613332488,
365
{'epoch': 9, 'Train_loss': 0.05414149537682533, 'Test_loss': 0.05413860082626343, 'Train_f1_score': 0.8628391591066702, 'Train_auroc': 0.9855724189810453, 'Test_f1_score': 0.6404338613332488, 'Test_auroc': 0.809254694122525}

# small 297
'Test_loss': 0.06896079331636429, 'Train_f1_score': 0.8573338092119004, 'Test_f1_score': 0.6255653200609244,
{'epoch': 10, 'Train_loss': 0.05524343252182007, 'Test_loss': 0.06896079331636429, 'Train_f1_score': 0.8573338092119004, 'Train_auroc': 0.9789479786623391, 'Test_f1_score': 0.6255653200609244, 'Test_auroc': 0.8263336407405123}

# big 356
0.6414172258900915 los 0.097764
{'epoch': 10, 'Train_loss': 0.0485670231282711, 'Test_loss': 0.09776438772678375, 'Train_f1_score': 0.8644148884290577, 'Train_auroc': 0.9822453926819152, 'Test_f1_score': 0.6414172258900915, 'Test_auroc': 0.823970368153288}

# tiny 
#160

loss <- c(0.101374,0.0689607,0.097, 0.0541)
size <- c(160,297,356,462)
#f1 <- c(0.62556,0.641417,0.64043,)
loss <- c(0.101374,0.0689607, 0.0541)
size <- c(160,297,462)

seg_results <- data.frame(cbind(loss,size))
seg_results

ggplot(seg_results, aes(x=size,y=loss)) + 
  geom_line() + labs(x="Training Set Size",y="Test Loss")
