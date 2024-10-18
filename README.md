# MP-GAE
Input:Temperature(batch_size,40,40);Humidity(batch_size,40,40);HFR(batch_size,40,40);density(batch_size,40,40);voltage(batch_size,40,40);operate_data(batch_size,40,8);operate_data_future(batch_size,40,1)
Output:Temperature_prediction(batch_size,1,40);Humidity_prediction(batch_size,1,40);HFR_prediction(batch_size,1,40);density_prediction(batch_size,1,40);voltage_prediction(batch_size,1,40)
operate_data include: density_load; temperature_water_in; an_RH; an_ER; an_P; ca_P; ca_RH; ca_ER.  This eight values remain the operate environment of PEMFC.
