The aim of this project is to develop a voice assistant. In the first step, we intend for our voice assistant to be able to identify 12 different categories. These categories include:

Stocks
Currency
Coins
Banks
Gold
Oil
Derivatives
Metals
Equity Funds
Fixed Income Funds
Mixed Funds
Tradable Funds

Assuming that our input only includes one of these 12 items, we will design a classifier with a size of less than 10 megabytes to accurately classify these 12 groups.

Finally, the output of this project will be a Recognizer class that will have two methods:

model_load method: This method should load the desired model into RAM memory upon invocation and be ready for use.
predict method: The input to this method will be the address of an audio file, and this method will be used to classify files.
