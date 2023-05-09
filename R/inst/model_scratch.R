
#plot_predictions(area_temp_sex, type="conditional", condition = list("wing_area_sc","temp_indv_sc"#,"Sex")) + xlab("Scaled wing area") + ylab("Proportion of black color cluster in wing") + ylim(0,0.4)

area_temp_sex_brown <- glmmTMB(brown ~ wing_area_sc + temp_indv_sc + Sex + (1 | species), data=wings2, family=beta_family(link="logit"),  ziformula = ~., na.action = "na.fail")
summary(area_temp_sex_brown)
plot_predictions(area_temp_sex_brown, type="conditional", condition = list("wing_area_sc","temp_indv_sc","Sex")) + xlab("Scaled wing area") + ylab("Proportion of black color cluster in wing") + ylim(0,0.4)

area_temp_sex_yellow <- glmmTMB(yellow ~ wing_area_sc + temp_indv_sc + Sex + (1 | species), data=wings2, family=beta_family(link="logit"),  ziformula = ~., na.action = "na.fail")
summary(area_temp_sex_yellow)
plot_predictions(area_temp_sex_yellow, type="conditional", condition = list("wing_area_sc","temp_indv_sc","Sex")) + xlab("Scaled wing area") + ylab("Proportion of black color cluster in wing") + ylim(0,0.4)


#area_temp_sex_log <- glmer(black01 ~ wing_area_sc * temp_indv_sc + Sex  + (1 | species), data=wings2,
#                         family="binomial")
#plot_model(area_temp_sex_glm, type="pred", c("wing_area_sc","temp_indv_sc","Sex"))

# sex flight temp
sex_flight_temp <- glmmTMB(black ~ Sex * flight_type + temp_indv_sc  + (1 | species), data=wings2, family=beta_family(link="logit"),  ziformula = ~., na.action = "na.fail")
summary(sex_flight_temp)

plot_predictions(sex_flight_temp, type="conditional", condition = list("flight_type", "Sex","temp_indv_sc"))

sex_flight_temp_log <- glmer(black01 ~ Sex * flight_type + temp_indv_sc  + (1 | species), data=wings2,
                             family="binomial")
plot_model(sex_flight_temp_log, type="pred", c("flight_type","temp_indv_sc","Sex"))

library(performance)
r2_nakagawa(sex_flight_temp_log)
r2_nakagawa(area_temp_sex_glm)
# strong set of effects tied to area '

# plot logistic

# plot conditional
library(marginaleffects)
plot_predictions(area_temp_sex, type="conditional", condition = list( "Area_total_sc","temp_indv_sc", "Sex"))
plot_predictions(sex_flight_temp, type="conditional", condition = list("flight_type", "Sex","temp_indv_sc"))

library(sjPlot)
plot_model(area_temp_sex, type="pred", c("Area_total_sc","temp_indv_sc", "Sex"))
plot_model(sex_flight_temp, type="pred", c("flight_type", "Sex","temp_indv_sc"))

# logistic regressions
