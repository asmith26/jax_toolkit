# What are some nice features of Matthews correlation coefficient vs F1 score?

## MCC can raise a "red flag" when many parts of the confusion matrix are 0
For example:
  
|         | Predicted: 0 | Predicted: 1 |
|---------|--------------|--------------|
| True: 0 | TN = 0       | FP = 5       |
| True: 1 | FN = 0       | TP = 95      |

yields:

<img src="https://latex.codecogs.com/svg.latex?\mbox{F1 score}=\frac{2 \times \mbox{TP}}{2\times\mbox{TP}+\mbox{FP}+\mbox{FN}} = \frac{2\times95}{2\times95 + 5} = 0.974" />

<img src="https://latex.codecogs.com/svg.latex?\mbox{MCC}=\frac{\mbox{TP}\times\mbox{TN}-\mbox{FP}\times\mbox{FN}}{\sqrt{(\mbox{TP}+\mbox{FP})(\mbox{TP}+\mbox{FN})(\mbox{TN}+\mbox{FP})(\mbox{TN}+\mbox{FN})}} = \frac{96\times0-5\times0}{\sqrt{100\times95\times5\times0}} = \mbox{undefined}" />

- F1 score is sensitive to which class is positive/negative. MCC isn't:

F1 score = 0.952, MCC = 0.135

|         | Predicted: 0 | Predicted: 1 |
|---------|--------------|--------------|
| True: 0 | TN = 1       | FP = 4       |
| True: 1 | FN = 5       | TP = 90      |

F1 score = 0.182, MCC = 0.135

|         | Predicted: 0 | Predicted: 1 |
|---------|--------------|--------------|
| True: 0 | TN = 90       | FP = 5       |
| True: 1 | FN = 4        | TP = 1       |

# Useful resources
[1] [Machine Learning Model Evaluation Metrics](https://www.youtube.com/watch?v=wpQiEHYkBys)
