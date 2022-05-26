#include <iostream>
#include <random>
#include <algorithm> // max_element
#include <cmath> // abs

class BanditProblem {
    private:
        int iterations, t, k;
        double epsilon;
        std::vector<double> q, q_a;
        std::vector<int> n;

        std::default_random_engine generator;

    public:
    
    // Constructor.
    BanditProblem(int k, double epsilon, int iterations){
        this->k = k;
        this->iterations = iterations;
        this->epsilon = epsilon;

        // Initialize n and q for k bandits/actions.
        n.assign(k, 0);
        q.assign(k, 0);  

        /* For this problem, each action a from 1 to k has an expected reward q*(a) =  E[R_t | A_t = a].
        R_t is the reward at timestep t, and A_t is the action taken at that timestep. To emulate a problem,
        q*(a) can be sampled from the Normal distrubtion. */
        
        std::normal_distribution<double> distribution(0, 1);
        q_a.assign(k, 0);
        for(int i=0; i<k; i++){
            q_a[i] = std::abs(distribution(generator));
            std::cout<<q_a[i]<<std::endl;
        }
        

    }

    void test(){
        /*for(t=0; t<iterations; t++){
            std::cout << "Iteration: " << t << "\n" << std::endl;
        }*/
        for(int i=0; i<k; i++){
            std::cout << "a, n and q: " << i << " " << n[i] << " " << q[i] << std::endl;
        }
        for(int i=0; i<k; i++){
            std::cout << "q*(a) values: a= " << i << " q*(a)=  " << q_a[i] << std::endl;
        } 
        // Check vector size.
        std::cout << "Size of n: " << n.size() << " Size of q: " << q.size() << std::endl;

        // Test bandit.
        for(t=0; t<iterations; t++){
            std::cout << "Iteration: " << t << ", Action: 3, R_t = "<< bandit(0) << std::endl;
        }
    }

    void run(std::vector<double> &avg_reward, std::vector<double> &avg_optimal_action){
        // Run the problem for specifed number of iterations.
        int a = 0; // the action chosen at timestep t.
        int best_a = 0; // the best possible action chosen at timestep t.
        double r[k] = {0}; // the rewards for choosing action a at timestep t.

        // Statistics for evaluation.
        bool optimal_a[iterations] = {false}; // True if the optimal action was taken at timestep t.
        double reward_at_t[iterations] = {0};

        // Uniform distribution for chance of exploration.
        std::uniform_real_distribution<double> uniform_explore(0,1);
        // Uniform distribution for selecting a random action.
        std::uniform_int_distribution<int> uniform_int(0, k-1); // a <= sample <= b, so need to sample up to k-1.

        int count_random = 0;
        int count_chosen = 0;

        for(int t=0; t<iterations; t++){
            // If probability < 1 - epsilon, use argmax Q(a) to select best action.
            if(uniform_explore(generator) < 1 - epsilon){
                a = std::distance(q.begin(), std::max_element(q.begin(), q.end()));
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
            best_a = std::distance(r, std::max_element(r, r + k));
            // If best_a is equal to a, then the best action was selected.
            if(best_a == a){
                optimal_a[t] = true;
            }
            // TODO: fix this.
            // Update avg_optimal_action using optimal a. This stores the current % accuracy the best action was taken at each t.
            if(t > 0){
                avg_optimal_action[t] += ((double)std::accumulate(optimal_a, optimal_a + t, 0))/t; //  Sum from first element up to current element at time t. Divide by t for accuracy.
                //std::cout<<"Test accumulate"<<std::endl;
                //std::cout<<((double)std::accumulate(optimal_a, optimal_a + t, 0))/t<<std::endl;
            } else {
                // Avoid division by 0 on first timestep. 
                avg_optimal_action[t] += std::accumulate(optimal_a, optimal_a + t, 0);
                
            }
            // Add r[a] to avg_reward for this timestep t.
            avg_reward[t] += r[a];
            //std::cout << r[a];

            
            // Only the reward r[a] of the chosen action a is used in updating the corresponding N(a) and Q(a) on this timestep.
            // Update "step-size" parameter (1/n) for sample-average method of estimating action-values.
            n[a] += 1;
            q[a] = q[a] + ((1/n[a]) * (r[a] - q[a]));

        }
        std::cout<<"Total correct actions taken: "<<std::accumulate(optimal_a, optimal_a + iterations, 0)<<std::endl;
        std::cout<<"Epsilon: "<<epsilon<<" Iterations: "<<iterations<<std::endl;
        std::cout<<"Total actions selected: "<<count_chosen<<std::endl;
        std::cout<<"Total random actions taken: "<<count_random<<std::endl;
        

        
    }

    double bandit(int action){
        // Sample from a normal distribution with mean = q*(a) for the chosen action a, to obtain reward for this step.
        std::normal_distribution<double> distribution(q_a[action], 1);
        return (double) std::abs(distribution(generator));
    }

};

void print_stats(std::vector<double> &avg_reward, std::vector<double> &avg_optimal_action){
    std::cout<<"Average reward over 1000 timesteps:"<< std::endl;
    for(int i=0; i < avg_reward.size(); i++)
        std::cout <<i<<": "<<avg_reward.at(i) <<std::endl;
    std::cout<<"Average optimal action accuracy over 1000 timesteps:"<< std::endl;
    for(int i=0; i < avg_optimal_action.size(); i++)
        std::cout <<i<<": "<<avg_optimal_action.at(i) <<std::endl;
}

int main(){

    // Number of experiments to perform.
    const int n_experiments = 2000;
    const int timesteps = 1000;
    // Run a single bandit problem.
    BanditProblem test_bandit = BanditProblem(10, 0.1, timesteps);
    // Input a reference to results vectors to be updated.
    std::vector<double> avg_reward, avg_optimal_action;
    avg_reward.assign(timesteps,0);
    avg_optimal_action.assign(timesteps,0);
    // Run bandit trials.
    test_bandit.run(avg_reward, avg_optimal_action);

    // Print out statistics.
    //print_stats(avg_reward, avg_optimal_action);


    return 0; 
}


 /*
    int v[10] = {1, 2, 3, 4 ,7, 5, 9, 6, 4, 2};
    std::vector<double> vec = {3.2, 4.6, 5.6, 1.2, 4.4};

    int arg_max = std::distance(v, std::max_element(v, v + 10));
    int vec_arg_max = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));

    std::cout << "Max array: " << arg_max <<std::endl;
    std::cout << "ArgMax vector: " << vec_arg_max <<std::endl;
    */