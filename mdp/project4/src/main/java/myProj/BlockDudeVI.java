package myProj;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.model.FullModel;
import burlap.mdp.singleagent.model.TransitionProb;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;

import java.util.*;


public class BlockDudeVI extends MDPSolver implements Planner, QProvider {

    protected Map<HashableState, Double> valueFunction;
    protected ValueFunction vinit;
    protected int numIterations;


    public BlockDudeVI(SADomain domain, double gamma,
                       HashableStateFactory hashingFactory, ValueFunction vinit, int numIterations){
        this.solverInit(domain, gamma, hashingFactory);
        this.vinit = vinit;
        this.numIterations = numIterations;
        this.valueFunction = new HashMap<HashableState, Double>();
    }

    @Override
    public double value(State s) {
        Double d = this.valueFunction.get(hashingFactory.hashState(s));
        if(d == null){
            return vinit.value(s);
        }
        return d;
    }

    @Override
    public List<QValue> qValues(State s) {
        List<Action> applicableActions = this.applicableActions(s);
        List<QValue> qs = new ArrayList<QValue>(applicableActions.size());
        for(Action a : applicableActions){
            qs.add(new QValue(s, a, this.qValue(s, a)));
        }

        return qs;
    }

    @Override
    public double qValue(State s, Action a) {

        if(this.model.terminal(s)){
            return 0.;
        }

        //what are the possible outcomes?
        List<TransitionProb> tps = ((FullModel)this.model).transitions(s, a);

        //aggregate over each possible outcome
        double q = 0.;
        for(TransitionProb tp : tps){
            //what is reward for this transition?
            double r = tp.eo.r;

            //what is the value for the next state?
            double vp = this.valueFunction.get(this.hashingFactory.hashState(tp.eo.op));

            //add contribution weighted by transition probability and
            //discounting the next state
            q += tp.p * (r + this.gamma * vp);
        }

        return q;
    }

    @Override
    public GreedyQPolicy planFromState(State initialState) {

        HashableState hashedInitialState = this.hashingFactory.hashState(initialState);
        if(this.valueFunction.containsKey(hashedInitialState)){
            return new GreedyQPolicy(this); //already performed planning here!
        }

        //if the state is new, then find all reachable states from it first
        this.performReachabilityFrom(initialState);

        //now perform multiple iterations over the whole state space
        for(int i = 0; i < this.numIterations; i++){
            //iterate over each state
            for(HashableState sh : this.valueFunction.keySet()){
                //update its value using the bellman equation
                this.valueFunction.put(sh, QProvider.Helper.maxQ(this, sh.s()));
            }
        }

        return new GreedyQPolicy(this);

    }

    @Override
    public void resetSolver() {
        this.valueFunction.clear();
    }

    public void performReachabilityFrom(State seedState){

        Set<HashableState> hashedStates = StateReachability.getReachableHashedStates(seedState, this.domain, this.hashingFactory);

        //initialize the value function for all states
        for(HashableState hs : hashedStates){
            if(!this.valueFunction.containsKey(hs)){
                this.valueFunction.put(hs, this.vinit.value(hs.s()));
            }
        }

    }


    public static void main(String [] args){

        BlockDude bd = new BlockDude();
        bd.setTf(new BlockDudeTF());
        SADomain domain = bd.generateDomain();

        //TerminalFunction tf = new BlockDudeTF();
        //RewardFunction rf =  new GoalBasedRF(new TFGoalCondition(tf), 5., -0.1);;

        State s = BlockDudeLevelConstructor.getLevel1(domain);


        //setup vi with 0.99 discount factor, a value
        //function initialization that initializes all states to value 0, and which will
        //run for 30 iterations over the state space
        BlockDudeVI vi = new BlockDudeVI(domain, 0.99, new SimpleHashableStateFactory(),
                new ConstantValueFunction(0.0), 30);

        //run planning from our initial state
        final long startTime = System.currentTimeMillis();
        Policy p = vi.planFromState(s);
        final long endTime = System.currentTimeMillis();

        //evaluate the policy with one roll out visualize the trajectory
        Episode ea = PolicyUtils.rollout(p, s, domain.getModel());

        double sum = 0;
        for(Double r : ea.rewardSequence) {
            sum += r;
        }

        System.out.println("Steps to exit: " + ea.actionSequence.size());
        System.out.println("Run time: " + (endTime - startTime) + " milliseconds");
        System.out.println("Reward Sum: " + sum);

        /*
        List<EpisodeAnalysis> lea = new LinkedList<EpisodeAnalysis>();
        lea.add(ea);

        Visualizer v = BlockDudeVisualizer.getVisualizer(30 ,30);
        new EpisodeSequenceVisualizer(v, domain, lea);
        */
    }

}