namespace SylvesterSolver.FSharp

open System

[<CLIMutable>]
type SolveResult =
    {
        Solution: double[,]
        ResidualNorm: double
        Iterations: int
        Converged: bool
        ResidualHistory: double[]
    }

module internal Matrix =
    let rows (matrix: double[,]) = matrix.GetLength(0)
    let cols (matrix: double[,]) = matrix.GetLength(1)

    let create rowsCount colsCount = Array2D.zeroCreate<double> rowsCount colsCount

    let copy (matrix: double[,]) = Array2D.copy matrix

    let identity size =
        let result = create size size

        for i in 0 .. size - 1 do
            result[i, i] <- 1.0

        result

    let transpose (matrix: double[,]) =
        let m = rows matrix
        let n = cols matrix
        let result = create n m

        for i in 0 .. m - 1 do
            for j in 0 .. n - 1 do
                result[j, i] <- matrix[i, j]

        result

    let addScaledIdentity (matrix: double[,]) scalar =
        let n = rows matrix
        let result = copy matrix

        for i in 0 .. n - 1 do
            result[i, i] <- result[i, i] + scalar

        result

    let subtractScaledIdentity (matrix: double[,]) scalar = addScaledIdentity matrix (-scalar)

    let multiply (left: double[,]) (right: double[,]) =
        let m = rows left
        let p = cols left
        let n = cols right
        let result = create m n

        for i in 0 .. m - 1 do
            for k in 0 .. p - 1 do
                let lv = left[i, k]

                for j in 0 .. n - 1 do
                    result[i, j] <- result[i, j] + lv * right[k, j]

        result

    let subtract (left: double[,]) (right: double[,]) =
        let m = rows left
        let n = cols left
        let result = create m n

        for i in 0 .. m - 1 do
            for j in 0 .. n - 1 do
                result[i, j] <- left[i, j] - right[i, j]

        result

    let add (left: double[,]) (right: double[,]) =
        let m = rows left
        let n = cols left
        let result = create m n

        for i in 0 .. m - 1 do
            for j in 0 .. n - 1 do
                result[i, j] <- left[i, j] + right[i, j]

        result

    let frobeniusNorm (matrix: double[,]) =
        let mutable sum = 0.0

        for i in 0 .. rows matrix - 1 do
            for j in 0 .. cols matrix - 1 do
                let v = matrix[i, j]
                sum <- sum + v * v

        sqrt sum

module internal LinearAlgebra =
    open Matrix

    let private swapRows (matrix: double[,]) first second =
        if first <> second then
            for j in 0 .. cols matrix - 1 do
                let tmp = matrix[first, j]
                matrix[first, j] <- matrix[second, j]
                matrix[second, j] <- tmp

    let private luDecompose (matrix: double[,]) =
        let n = rows matrix
        let lu = copy matrix
        let pivots = Array.init n id

        for k in 0 .. n - 1 do
            let mutable pivotRow = k
            let mutable pivotValue = abs lu[k, k]

            for i in k + 1 .. n - 1 do
                let candidate = abs lu[i, k]

                if candidate > pivotValue then
                    pivotValue <- candidate
                    pivotRow <- i

            if pivotValue < 1e-14 then
                invalidArg "matrix" "Matrix is singular or ill-conditioned for LU decomposition."

            if pivotRow <> k then
                swapRows lu pivotRow k
                let tmp = pivots[pivotRow]
                pivots[pivotRow] <- pivots[k]
                pivots[k] <- tmp

            for i in k + 1 .. n - 1 do
                lu[i, k] <- lu[i, k] / lu[k, k]

                for j in k + 1 .. n - 1 do
                    lu[i, j] <- lu[i, j] - lu[i, k] * lu[k, j]

        lu, pivots

    let solveLeft (matrix: double[,]) (rhs: double[,]) =
        let n = rows matrix

        if cols matrix <> n then
            invalidArg "matrix" "Coefficient matrix must be square."

        if rows rhs <> n then
            invalidArg "rhs" "RHS row count must match coefficient matrix size."

        let lu, pivots = luDecompose matrix
        let x = copy rhs

        for i in 0 .. n - 1 do
            let p = pivots[i]

            if p <> i then
                swapRows x i p

        for i in 0 .. n - 1 do
            for j in 0 .. cols x - 1 do
                for k in 0 .. i - 1 do
                    x[i, j] <- x[i, j] - lu[i, k] * x[k, j]

        for i in n - 1 .. -1 .. 0 do
            for j in 0 .. cols x - 1 do
                for k in i + 1 .. n - 1 do
                    x[i, j] <- x[i, j] - lu[i, k] * x[k, j]

                x[i, j] <- x[i, j] / lu[i, i]

        x

    let solveRight (matrix: double[,]) (rhs: double[,]) =
        let rhsT = transpose rhs
        let solved = solveLeft (transpose matrix) rhsT
        transpose solved

type SylvesterAdiSolver private () =
    static member private Residual(a: double[,], b: double[,], c: double[,], x: double[,]) =
        let ax = Matrix.multiply a x
        let xb = Matrix.multiply x b
        let lhs = Matrix.add ax xb
        let residual = Matrix.subtract lhs c
        Matrix.frobeniusNorm residual

    static member private ValidateInputs(a: double[,], b: double[,], c: double[,], pShifts: double[], qShifts: double[]) =
        if isNull a || isNull b || isNull c then
            nullArg "a/b/c"

        if isNull pShifts || isNull qShifts then
            nullArg "pShifts/qShifts"

        let m = Matrix.rows a
        let n = Matrix.rows b

        if Matrix.cols a <> m then
            invalidArg "a" "Matrix A must be square."

        if Matrix.cols b <> n then
            invalidArg "b" "Matrix B must be square."

        if Matrix.rows c <> m || Matrix.cols c <> n then
            invalidArg "c" "Matrix C shape must be m x n, where A is m x m and B is n x n."

        if pShifts.Length = 0 || qShifts.Length = 0 then
            invalidArg "pShifts/qShifts" "Shift sequences cannot be empty."

    static member Solve
        (
            a: double[,],
            b: double[,],
            c: double[,],
            pShifts: double[],
            qShifts: double[],
            tolerance: double,
            maxIterations: int
        )
        : SolveResult =
        SylvesterAdiSolver.ValidateInputs(a, b, c, pShifts, qShifts)

        if tolerance <= 0.0 then
            invalidArg "tolerance" "Tolerance must be positive."

        if maxIterations <= 0 then
            invalidArg "maxIterations" "Max iterations must be positive."

        let mutable x = Matrix.create (Matrix.rows a) (Matrix.cols b)
        let residuals = ResizeArray<double>()
        let mutable converged = false
        let mutable iterations = 0
        let mutable residual = Double.PositiveInfinity

        while not converged && iterations < maxIterations do
            let p = pShifts[iterations % pShifts.Length]
            let q = qShifts[iterations % qShifts.Length]

            let leftHalf = Matrix.addScaledIdentity a p
            let rightHalf = Matrix.subtract c (Matrix.multiply x (Matrix.subtractScaledIdentity b p))
            let xHalf = LinearAlgebra.solveLeft leftHalf rightHalf

            let rightFull = Matrix.addScaledIdentity b q
            let leftFullRhs = Matrix.subtract c (Matrix.multiply (Matrix.subtractScaledIdentity a q) xHalf)
            x <- LinearAlgebra.solveRight rightFull leftFullRhs

            residual <- SylvesterAdiSolver.Residual(a, b, c, x)
            residuals.Add(residual)
            iterations <- iterations + 1
            converged <- residual < tolerance

        {
            Solution = x
            ResidualNorm = residual
            Iterations = iterations
            Converged = converged
            ResidualHistory = residuals.ToArray()
        }
