IDK (I Don’t Know) cascades have been presented as an alternative to current classification models. These structures comprise of a "cascade" of classifiers that only categorizes an input if a classifier outputs a confidence level that exceeds a predetermined threshold. If it does not, it outputs the class "I don’t know" and moves on to the next classifier in the cascade. This process repeats until a classification is made, or the last classifier is hit. This final classifier will make a classification regardless of the outputted confidence threshold.
![IDK Cascade Image](https://github.com/user-attachments/assets/a91343c0-6cf6-4909-892d-03cd2ec70dea)

_A Diagram of a working IDK Cascade_

Dynamic IDK cascades implement a minimum threshold on the first classifier. It assumes that if the classifiers are working in tandem, and have some similarities between each other, the confidence of the first classifier can accurately predict whether the second classifier will output "I Don't Know" or an actual class. By implementing a minimum threshold that skips the second classifier if it is not reached, we save time.
![IDK Cascade Image 2](https://github.com/user-attachments/assets/b53f1489-3740-4153-8ab9-f44603fac071)
