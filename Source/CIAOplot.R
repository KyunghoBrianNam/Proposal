# Create CAIO plot from x.
# x should be a data frame defining the indices and values of an
# upper triangular matrix where the first column consists of the
# first indices, the second column consists of the second
# indices, and the third column consists of the values.
CAIOplot <- function(
        x,
        na.rm = TRUE,
        col = gray.colors(50, start = 0.0, end = 0.95, gamma = 2.2, alpha = NULL),
        margins = c(4, 4),
        main = NULL,
        xlab = NULL,
        ylab = NULL) {
    # Compute matrix diminsions from number of elements in the
    # upper triangular matrix.
    n <- (sqrt(8 * nrow(x) + 1) - 1) / 2

    # Create matrices from data frame
    M <- matrix(NA, nrow = n, ncol = n)
    for (i in 1:nrow(x)) {
        M[x[i,1], x[i,2]] = x[i,3]
    }

   # Condition matrix assuming maximization problem
   N <- max(M, na.rm = na.rm) - M
   NMin <- min(N, na.rm = na.rm)
   NMax <- max(N, na.rm = na.rm)
   M <- (N - NMin) / (NMax - NMin)

   # Generate CIAO plot
   heatmap(M, Rowv = NA, Colv = NA, scale = "none", na.rm = na.rm, col = col, margins = margins, main = main, xlab = xlab, ylab = ylab)
}
