def get_opt_coeff():
    # hard-coded the optimization coefficients from Dr.Grizzle's book
    # link: https://web.eecs.umich.edu/~grizzle/biped_book_web/
    # at page 14 using minimum energy cost optimization
    alpha = [0.512, 0.073, 0.035, -0.819, -2.27, 3.26, 3.11, 1.89]

    return alpha