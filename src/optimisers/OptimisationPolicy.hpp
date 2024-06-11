#ifndef OPTIMISATIONPOLICY_HPP
#define OPTIMISATIONPOLICY_HPP

#include <vector>

class OptimisationPolicy {

public:
    void SetInnerLoopTimeAllocation(double ms);
    void SetMinInnerLoopEvals(int it);

    void Inform(int xiCount, double weightMatrixCreationTimems);
    void Inform(int xiCount, int innerEvals, double tookms);

    bool KnowMaxMeritSamplesToUse() const;
    int GetMaxSamplesToUse() const;

    bool KnowEvalsToDo(int xiCount) const;
    int GetInnerLoopEvalsToPerform(int xiCount) const;

private:
    double innerLoopTimeAllocationms;
    int minInnerLoopEvals;

    int maxMeritSamplesToUse;
    std::vector<int> EvalsToPerformPerxiCount;
};

#endif
