using System;

namespace SparseCourseworkRunner;

public interface IPreconditioner
{
    void Apply(ReadOnlySpan<double> rhs, Span<double> destination);
}
