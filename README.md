graph TD
    %% Styles
    classDef current fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef future fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 5 5,color:#856404;
    classDef merge fill:#e2e3e5,stroke:#383d41,stroke-width:2px,color:#383d41;
    classDef user fill:#cce5ff,stroke:#004085,stroke-width:2px,color:#004085;

    %% User Layer (Future)
    subgraph UserLayer [User Preferences & Constraints]
        U1(Risk Aversion / Gut Feeling) ::: user
        U2(Insider Insights / Discounts) ::: user
    end

    %% The Mixing Engine (Convergence Point)
    subgraph Mixing [Decision Engine]
        MIX{Adaptive Opinion Mixing} ::: future
        NOTE[Method: Robust Regression / Bayesian Dirichlet] ::: future
        SQP(Constrained Sequential <br/>Quadratic Programming) ::: current
        ALLOC([Final Optimized Allocation]) ::: merge
    end

    %% Branch 1: Technical
    subgraph Tech [Technical Branch]
        direction TB
        ARX(Linear Auto-Regression <br/> w/ Exogenous Variables) ::: current
        GA(Genetic Algorithm <br/> Structure Estimation) ::: current
        FORGET(Optimized Forgetting) ::: current
        
        %% Future Tech
        GMM(Mixed Gaussian Models) ::: future
        NLP(NLP Sentiment Analysis) ::: future
        
        %% Tech Flow
        GA --> ARX
        FORGET --> ARX
        ARX --> SQP
        GMM -.-> MIX
        NLP -.-> MIX
    end

    %% Branch 2: Fundamental
    subgraph Fund [Fundamental Branch]
        direction TB
        VAL(Company Valuation Ranking) ::: current
        EXP(Expert Opinions) ::: future
        METRIC(Success Metrics & <br/> Param Optimization) ::: future
        
        %% Fund Flow
        VAL -.-> METRIC
        METRIC -.-> MIX
        EXP -.-> MIX
    end

    %% Main Connections
    SQP --> MIX
    MIX --> ALLOC
    UserLayer -.-> ALLOC

    %% Legend
    linkStyle default stroke-width:2px,fill:none,stroke:#333;
