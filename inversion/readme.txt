For inversion, I used over 200k requests whose results were spread over 10+ DBs
The best inversion results visually were from the secondbig.db which was mostly made with perturbed cifar images (left a few other inversion samples from other dbs for comparison in the results folder).

The pipeline is straightforward - generate the data, train the models, and print the images retrieved by the inversion attack.
