#include <iostream>
#include <random>
#include <algorithm> // max_element
#include <cmath> // abs
#include <vector>
#include <iomanip>

using namespace std;

struct BanditResults {
        double epsilon;
        vector<double> optimal_action;
        vector<double> reward_log;
    };

class BanditProblem {
    private:
        int iterations, t, k;
        double epsilon;
        vector<double> q, q_a;
        vector<int> n;

        default_random_engine generator;
        vector<normal_distribution<double>> reward_distributions;

    public:
    
    // Constructor.
    BanditProblem(int k, double epsilon, int iterations, unsigned int seed = 0){
        this->k = k;
        this->iterations = iterations;
        this->epsilon = epsilon;

        // Initialize n and q for k bandits/actions.
        n.assign(k, 0);
        q.assign(k, 0);  

        /* For this problem, each action a from 1 to k has an expected reward q*(a) =  E[R_t | A_t = a].
        R_t is the reward at timestep t, and A_t is the action taken at that timestep. To emulate a problem,
        q*(a) can be sampled from the Normal distrubtion. */
        generator.seed(seed);
        normal_distribution<double> distribution(0, 1);
        q_a.assign(k, 0);
        for(int i=0; i<k; i++){
            q_a[i] = distribution(generator);
            reward_distributions.push_back(normal_distribution<double>{q_a[i], 1});
        }        

    }

    BanditResults run(){
        // Run the problem for specifed number of iterations.
        int a = 0; // the action chosen at timestep t.
        int best_a = 0; // the best possible action chosen at timestep t.
        double r[k] = {0}; // the rewards for choosing action a at timestep t.

        // Statistics for evaluation.
        vector<double> reward_log(iterations, 0);
        vector<double>  optimal_action(iterations, 0);

        bool optimal_a[iterations] = {false}; // True if the optimal action was taken at timestep t.

        // Uniform distribution for chance of exploration.
        uniform_real_distribution<double> uniform_explore(0,1);
        // Uniform distribution for selecting a random action.
        uniform_int_distribution<int> uniform_int(0, k-1); // a <= sample <= b, so need to sample up to k-1.

        int count_random = 0;
        int count_chosen = 0;

        for(int t=0; t<iterations; t++){
            // If probability < 1 - epsilon, use argmax Q(a) to select best action.
            if(uniform_explore(generator) < 1.0 - epsilon && t > 0){
                a = distance(q.begin(), max_element(q.begin(), q.end()));
                count_chosen += 1;

            } else {
                // Select a random action.
                a = uniform_int(generator);
                count_random += 1;
            }

            // Obtain rewards for all actions at timestep t - for the purpose of evaluating whether the best action was taken.
            for(int i = 0; i < k; i++){
                r[i] = bandit(i);
            }

            // Find best action in r - the argmax.
            best_a = distance(r, max_element(r, r + k));
            //DEBUG:
            //std::cout<<"Action taken: "<<a<<" Reward: "<<r[a];
            //std::cout<<" Optimal action: "<<best_a<<" Reward: "<<(*std::max_element(r, r + k))<<std::endl;

            // If best_a is equal to a, then the best action was selected.
            if(best_a == a){
                optimal_a[t] = true;
            }
            // TODO: fix this.
            // Update avg_optimal_action using optimal a. This stores the current % accuracy the best action was taken at each t.
            if(t > 0){
                optimal_action[t] = ((double) accumulate(optimal_a, optimal_a + t, 0)) * 100/t; //  Sum from first element up to current element at time t. Divide by t for accuracy.
            } else {
                // Avoid division by 0 on first timestep. 
                optimal_action[t] = ((double) accumulate(optimal_a, optimal_a + t, 0)) * 100;
                
            }
            // Add r[a] to avg_reward for this timestep t.
            reward_log[t] = r[a];
            

            
            // Only the reward r[a] of the chosen action a is used in updating the corresponding N(a) and Q(a) on this timestep.
            // Update "step-size" parameter (1/n) for sample-average method of estimating action-values.
            n[a] += 1;
            q[a] = q[a] + ((1.0/n[a]) * (r[a] - q[a]));
        }
        
        //cout<<"Total correct actions taken: "<<accumulate(optimal_a, optimal_a + iterations, 0)<<endl;
        //cout<<"Epsilon: "<<epsilon<<" Iterations: "<<iterations<<endl;
        //cout<<"Total actions selected: "<<count_chosen<<endl;
        //cout<<"Total random actions taken: "<<count_random<<endl;
        
        BanditResults result;
        result.optimal_action = optimal_action;
        result.reward_log = reward_log;
        result.epsilon = epsilon;
        return result;
        
    }

    double bandit(int action){
        // Sample from a normal distribution with mean = q*(a) for the chosen action a, to obtain reward for this step.
        //std::normal_distribution<double> distribution(q_a[action], 1);
        //return (double) distribution(generator);
        return reward_distributions[action](generator);
    }

};

void print_stats(vector<double> &avg_reward, vector<double> &avg_optimal_action){
    cout<<"Average reward over 1000 timesteps:"<<endl;
    for(int i=0; i < (int) avg_reward.size(); i++)
        cout <<i<<": "<<avg_reward.at(i) <<endl;

    cout<<"Average optimal action accuracy over 1000 timesteps:"<<endl;
    for(int i=0; i < (int) avg_optimal_action.size(); i++)
        cout <<i<<": "<<avg_optimal_action.at(i) <<"\%"<<endl;
}

void print_line(){
    for(int i=0; i<40; i++){
        cout<<"--";
    }
    cout<<endl;
}

// Prints progress of averaged results every i iterations.
void print_tabular(vector<BanditResults> result_vector, int k, int interval=100, int timesteps=1000, int n_experiments=2000){
    print_line();
    cout<<"Average reward over "<<n_experiments<<" runs using k = "<<k<<" and "<<timesteps<<" timesteps:"<<endl;
    print_line();
    cout<<"timestep";
    for(int i=0; i< (int) result_vector.size(); i++){
        // Print columns using epsilon values.
        cout<<"\t\t"<<result_vector[i].epsilon;
    }
    cout<<endl;
    print_line();
    // Print results of each test for current timestep.
    for(int i=0; i <= timesteps-1; i+=interval){
        cout<<"  "<<i<<"\t";
        for(int j=0; j< (int) result_vector.size(); j++){
            cout<<"\t\t"<<result_vector[j].reward_log[i];
        }
        cout<<endl;
    }

    cout<<endl;
    print_line();
    cout<<"Average optimal action accuracy over "<<n_experiments<<" runs using k = "<<k<<" and "<<timesteps<<" timesteps:"<<endl;
    print_line();
    cout<<"timestep";
    for(int i=0; i< (int) result_vector.size(); i++){
        // Print columns using epsilon values.
        cout<<"\t\t"<<result_vector[i].epsilon;
    }
    cout<<endl;
    print_line();
    // Print results of each test for current timestep.
    for(int i=0; i <= timesteps-1; i+=interval){
        cout<<"  "<<i<<"\t";
        for(int j=0; j< (int) result_vector.size(); j++){
            cout<<"\t\t"<<result_vector[j].optimal_action[i]<<"\%";
        }
        cout<<endl;
    }
}

// Runs all n_experiments and returns the averaged results.
BanditResults run_experiment_config(int k, double epsilon, int timesteps, int n_experiments){
    // Vectors to tally average results.
    vector<double> avg_reward(timesteps,0), avg_optimal_action(timesteps,0);
    
    BanditResults result;
    // Run bandit trials.
    for(int n=0; n<n_experiments; n++){
        //cout<<"Trial "<<n<<endl;
        result = BanditProblem(k, epsilon, timesteps, n).run();
        // Add current results to the total.
        for(int i=0; i<timesteps; i++){
            avg_optimal_action[i] += result.optimal_action[i];
            avg_reward[i] += result.reward_log[i];
        }
    }
    // Average results by dividing by n_experiments.
    // First two parameters give the range of elements to transform.
    // Third parameter is the intial iterator to where to store values after transform.
    // Fourth is the lambda function: takes n_experiments as a "captured" variable from the enclosing scope;
    // takes a double  reference as a parameter (each vector element). Followed by the function, returning the division.
    transform(avg_optimal_action.begin(), avg_optimal_action.end(), avg_optimal_action.begin(), [n_experiments](double &c){ return c/n_experiments; });
    transform(avg_reward.begin(), avg_reward.end(), avg_reward.begin(), [n_experiments](double &c){ return c/n_experiments; });
        
    BanditResults avg_result;
    avg_result.optimal_action = avg_optimal_action;
    avg_result.reward_log = avg_reward;
    avg_result.epsilon = epsilon;
    return avg_result;
  
}

int main(){

    // Set print precision.
    cout<<setprecision(4);
    // Number of experiments to perform.
    const int n_experiments = 2000;
    const int timesteps = 1001;
    const int k = 10;

    // Run a single bandit problem.
    //BanditProblem test_bandit = BanditProblem(k, 0.1, timesteps);
    //BanditResults test_result = test_bandit.run();
    // Print out statistics.
    //print_stats(test_result.reward_log, test_result.optimal_action);

    // Peform trial experiments.
    BanditResults greedy = run_experiment_config(k, 0, timesteps, n_experiments);
    BanditResults small_epsilon = run_experiment_config(k, 0.01, timesteps, n_experiments);
    BanditResults large_epsilon = run_experiment_config(k, 0.1, timesteps, n_experiments);
        
    vector<BanditResults> result_vector = {greedy, small_epsilon, large_epsilon};    
    print_tabular(result_vector, k, 100, timesteps, n_experiments);
    //print_stats(avg_reward, avg_optimal_action);

    return 0; 
}
