Arka Sadhu (140070011)
CS 725 : Machine Learning
Assignment 1
==============================
Organization of the directory
==============================
The directory is structured as follows:
template.py is the main file. It imports the model class from sgd_solver.py
sgd_solver contains all the optimization algorithms implemented in it.
final_weights.pkl contains the best weight in pickle format.
output.csv contains the output corresponding to these weights.
readme.txt is this file

====================
Gradient Descent :
====================
For p = 2:
	We use direct solution for p = 2 which is
	w = ((phi.T * phi) + lambda * I)^(-1) * (phi.T * y)

	We get this by direct differentiation.

For p = 1:
	For p = 1 we don't use gradient descent because it is not defined at some points.
	Therefore we use Iiterative Shrinkage Thresholding Algorithm (ISTA).
	It involves taking a function which is a majorizer of the original function and at the same time
	differentiable. And then we use a soft thresholding.
	The majorizer function is as follows :
	M(w) = ||y - phi * w||^2 + (w - w_k).T * (alpha * I - phi.T * phi) (w - w_k) + lambda ||w||_1
	We choose alpha to be greater than the maximum eigen value of (phi.T phi)
	On differentiating M(w) w.r.t w and setting it to 0, we get the corresponding w as
	w_(k+1) = soft(w_k + phi.T (y - phi w_k)/alpha; lambda/(2*alpha))
	where soft is the soft function given by
	soft(y; h) = y - h if y>=h, y + h if y <= -h, and 0 if -h <= y <= h

	We iterate this till convergence.
	The convergence criteria is set as whenever the weight updates is too small
	This is characterized by np.allclose(prev_weights, curr_weights)


For 1 < p < 2:
	We use stochastic gradient descent with either fixed iterations or going upto convergence

	What we do is initialize the weights assuming that p=2. It is experimentally found that doing this
	gives faster convergence rate.

	Then we start using the stochastic gradient descent algorithm (SGD)
	For SGD we take only one point at a time and update the weights.
	The learning rate is initially set to a constant, but then it is increased or decreased
	corresponding to the losses observed.

	Instead of observing the losses at every iteration to change the learning rate, we instead
	see this at every batch size, which is kept as an input parameter.
	If it is seen that the new loss is much lesser than the previous loss (a gap of 10 is chosen after experimentation),
	we increase the learning rate by a factor of 1.1 until the next batch size.

	If instead the new loss is much larger than the previous loss (again a gap of 10 is chosen after experimentation),
	we keep a counter for this, and decrease the learning rate by this counter.
	This is important because we don't want to decrease the learning rate in a geometric progression
	We rather keep it as initial_rate/counter, so it is a harmonic series essentially and therefore
	the sum doesn't die down in the limit.

	This is iterated till convergence which is again defined to be np.allclose(prev_weights, curr_weights) or
	if it has exceeded the maximum number of it

==================================================
Feature Engineering : Creation of the phi matrix
==================================================

All the given features are used except for the id field.
For the date field only the month, day and hour have been taken into consideration.
All other fields are taken in a linear fashion.
This gives us a 27 dim vector. A constant 1 is appended to it to make it 28 dim vector.
This corresponds to the bias term in the weight vector.
