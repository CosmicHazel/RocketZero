// C++11

#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <cassert>

#ifdef _WIN32
#include "..\..\common_lib\utils.cpp"
#else
#include "../../common_lib/utils.cpp"
#endif

/*
Rocket League Bot Overview:

This code is part of a Rocket League bot implementation using a novel "x-hot" control mechanism.
Key features of the bot:

1. x-Hot Control: A single AI agent simultaneously controls all NUM_ACTION_HEADS players on the same team.
   This allows for coordinated team strategies and complex multi-player maneuvers.

2. Large Action Space: Each individual player has ACTIONS_PER_PLAYER possible actions, resulting in a
   total action space of TOTAL_ACTIONS (ACTIONS_PER_PLAYER * NUM_ACTION_HEADS) for the entire team controlled by the AI.

3. MCTS Implementation: This file (cnode.cpp) implements the Monte Carlo Tree Search (MCTS)
   algorithm, which is used for decision making and planning in the bot's AI.

This unique approach allows the bot to make strategic decisions that consider the
actions of all team members simultaneously, potentially leading to more coordinated
and effective gameplay strategies.
*/

namespace tree
{

    CSearchResults::CSearchResults()
    {
        /*
        Overview:
            Initialization of CSearchResults, the default result number is set to 0.
        */
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num)
    {
        /*
        Overview:
            Initialization of CSearchResults with result number.
        Arguments:
            - num: The number of results to initialize.
        */
        this->num = num;
        this->search_paths.resize(num);  // Initialize search_paths first
        for (int i = 0; i < num; ++i) {
            this->search_paths[i] = std::vector<CNode*>();  // Initialize each inner vector
        }
        this->latent_state_index_in_search_path.resize(num);
        this->latent_state_index_in_batch.resize(num);
        this->last_actions.resize(num);
        this->search_lens.resize(num);
        this->virtual_to_play_batchs.resize(num);
        this->nodes.resize(num);
    }

    CSearchResults::~CSearchResults() {}

    //*********************************************************

    CNode::CNode()
    {
        /*
        Overview:
            Initialization of CNode.
        */
        this->prior = 0;
        this->legal_actions = legal_actions;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = {-1, -1, -1, -1};
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
    }

    CNode::CNode(float prior, std::vector<int> &legal_actions)
    {

        /*
        Overview:
            Initialization of CNode with prior value and legal actions.
        Arguments:
            - prior: the prior value of this node.
            - legal_actions: a vector of legal actions of this node.
        */

        //std::cout << "Entering CNode constructor" << std::endl;
        //std::cout << "prior: " << prior << std::endl;
        //std::cout << "legal_actions size: " << legal_actions.size() << std::endl;
        try
        {
            //std::cout << "Setting prior" << std::endl;
            this->prior = prior;
            //std::cout << "Copying legal_actions" << std::endl;
            this->legal_actions = legal_actions;

            //std::cout << "Initializing other members" << std::endl;
            this->is_reset = 0;
            this->visit_count = 0;
            this->value_sum = 0;
            this->best_action = {-1, -1, -1, -1};
            this->to_play = 0;
            this->value_prefix = 0.0;
            this->parent_value_prefix = 0.0;
            this->current_latent_state_index = -1;
            this->batch_index = -1;

            //std::cout << "CNode constructor completed successfully" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in CNode constructor: " << e.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown exception in CNode constructor" << std::endl;
            throw;
        }
    }

    CNode::~CNode() {}

    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float value_prefix, const std::vector<float> &policy_logits)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play: which player to play the game in the current node.
            - current_latent_state_index: the x/first index of hidden state vector of the current node, i.e. the search depth.
            - batch_index: the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - value_prefix: the value prefix of the current node.
            - policy_logits: the policy logit of the child nodes.
        */
        this->to_play = to_play;
        this->current_latent_state_index = current_latent_state_index;
        this->batch_index = batch_index;
        this->value_prefix = value_prefix;

        int action_num = policy_logits.size(); // Assuming policy_logits contains logits for all NUM_ACTION_HEADS players
        if (this->legal_actions.size() == 0)
        {
            for (int i = 0; i < action_num; ++i)
            {
                this->legal_actions.push_back(i);
            }
        }
        float temp_policy;
        float policy_sum = 0.0;

#ifdef _WIN32
        // 创建动态数组
        float *policy = new float[action_num];
#else
        float policy[action_num];
#endif

        float policy_max = FLOAT_MIN;
        for (auto a : this->legal_actions)
        {
            if (policy_max < policy_logits[a])
            {
                policy_max = policy_logits[a];
            }
        }

        for (auto a : this->legal_actions)
        {
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        for (auto a : this->legal_actions)
        {
            prior = policy[a] / policy_sum;
            std::vector<int> tmp_empty;
            this->children[a] = new CNode(prior, tmp_empty);
        }
#ifdef _WIN32
        // 释放数组内存
        delete[] policy;
#else
#endif
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises)
    {
        /*
        Overview:
            Add a noise to the prior of the child nodes.
        Arguments:
            - exploration_fraction: the fraction to add noise.
            - noises: the vector of noises added to each child node.
        */
        float noise, prior;
        for (int i = 0; i < this->legal_actions.size(); ++i)
        {
            noise = noises[i];
            CNode *child = this->get_child(this->legal_actions[i]);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::compute_mean_q(int isRoot, float parent_q, float discount_factor)
    {
        /*
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - isRoot: whether the current node is a root node.
            - parent_q: the q value of the parent node.
            - discount_factor: the discount_factor of reward.
        */
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        float parent_value_prefix = this->value_prefix;
        for (int i = 0; i < NUM_ACTION_HEADS; i++)
        {
            for (auto a : this->legal_actions)
            {
                CNode *child = this->get_child(a + i * ACTIONS_PER_PLAYER); // This is a simplification
                if (child->visit_count > 0)
                {
                    float true_reward = child->value_prefix - parent_value_prefix;
                    if (this->is_reset == 1)
                    {
                        true_reward = child->value_prefix;
                    }
                    float qsa = true_reward + discount_factor * child->value();
                    total_unsigned_q += qsa;
                    total_visits += 1;
                }
            }
        }

        float mean_q = 0.0;
        if (isRoot && total_visits > 0)
        {
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else
        {
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    int CNode::expanded()
    {
        /*
        Overview:
            Return whether the current node is expanded.
        */
        return this->children.size() > 0;
    }

    float CNode::value()
    {
        /*
        Overview:
            Return the estimated value of the current tree.
        */
        float true_value = 0.0;
        if (this->visit_count == 0)
        {
            return true_value;
        }
        else
        {
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<std::vector<int>> CNode::get_trajectory()
    {
        /*
        Overview:
            Find the current best trajectory starts from the current node.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from this node.
        */
        std::vector<std::vector<int>> traj;

        CNode *node = this;
        std::vector<int> best_action = node->best_action;
        while (best_action[0] >= 0)
        {
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> CNode::get_children_distribution()
    {
        /*
        Overview:
            Get the distribution of child nodes in the format of visit_count.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<int> distribution;
        if (this->expanded())
        {
            for (auto a : this->legal_actions)
            {
                CNode *child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode *CNode::get_child(std::vector<int> actions)
    {
        /*
        Overview:
            Get the child node corresponding to the input x-hot action vector.
        Arguments:
            - actions: vector of actions, one for each player (x-hot encoding)
        Returns:
            - child node pointer or nullptr if not found
        */
        if (actions.size() != NUM_ACTION_HEADS)
            return nullptr;

        uint64_t action_key = encode_action(actions);
        if (action_key >= children.size() || children[action_key] == nullptr)
            return nullptr;
        return children[action_key];
    }

    uint64_t CNode::encode_action(std::vector<int> actions)
    {
        /*
        Overview:
            Encode x-hot action vector into a single uint64_t key.
        Arguments:
            - actions: vector of actions, one for each player
        Returns:
            - encoded action key
        */
        uint64_t encoded = 0;
        for (size_t i = 0; i < NUM_ACTION_HEADS && i < actions.size(); ++i)
        {
            if (actions[i] >= 0 && actions[i] < ACTIONS_PER_PLAYER)
            {
                encoded += static_cast<uint64_t>(actions[i]) + i * ACTIONS_PER_PLAYER;
            }
        }
        return encoded < TOTAL_ACTIONS ? encoded : TOTAL_ACTIONS - 1;
    }

    CNode *CNode::get_child(uint64_t action)
    {
        /*
        Overview:
            Get the child node corresponding to the encoded action key.
        Arguments:
            - action: encoded action key
        Returns:
            - child node pointer or nullptr if not found
        */
        if (action >= children.size() || children[action] == nullptr)
            return nullptr;
        return children[action];
    }

    //*********************************************************

    CRoots::CRoots()
    {
        /*
        Overview:
            The initialization of CRoots.
        */
        this->root_num = 0;
    }

    CRoots::CRoots(int root_num, std::vector<std::vector<int>> &legal_actions_list)
    {
        std::cout << "Entering CRoots constructor" << std::endl;
        std::cout << "root_num: " << root_num << std::endl;
        std::cout << "legal_actions_list size: " << legal_actions_list.size() << std::endl;

        try
        {
            this->root_num = root_num;
            this->legal_actions_list = legal_actions_list;

            std::cout << "About to create roots vector" << std::endl;
            for (int i = 0; i < root_num; ++i)
            {
                std::cout << "Creating root " << i << std::endl;
                this->roots.push_back(CNode(0, this->legal_actions_list[i]));
                std::cout << "Root " << i << " created" << std::endl;
            }
            std::cout << "All roots created successfully" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in CRoots constructor: " << e.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown exception in CRoots constructor" << std::endl;
            throw;
        }

        // /*
        // Overview:
        //     The initialization of CRoots with root num and legal action lists.
        // Arguments:
        //     - root_num: the number of the current root.
        //     - legal_action_list: the vector of the legal action of this root.
        // */
        // this->root_num = root_num;
        // this->legal_actions_list = legal_actions_list;

        // for (int i = 0; i < root_num; ++i)
        // {
        //     this->roots.push_back(CNode(0, this->legal_actions_list[i]));
        // }
    }

    CRoots::~CRoots() {}

    void CRoots::prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].add_exploration_noise(root_noise_weight, noises[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots without noise.
        Arguments:
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear()
    {
        /*
        Overview:
            Clear the roots vector.
        */
        this->roots.clear();
    }

    std::vector<std::vector<std::vector<int>>> CRoots::get_trajectories()
    {
        /*
        Overview:
            Find the current best trajectory starts from each root.
        Returns:
            - trajs: a vector of trajectories, where each trajectory is a vector of x-hot actions.
        */
        std::vector<std::vector<std::vector<int>>> trajs;
        trajs.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions()
    {
        /*
        Overview:
            Get the children distribution of each root.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values()
    {
        /*
        Overview:
            Return the estimated value of each root.
        */
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    //*********************************************************
    //
    void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players)
    {
        /*
        Overview:
            Update the q value of the root and its child nodes.
        Arguments:
            - root: the root that update q value from.
            - min_max_stats: a tool used to min-max normalize the q value.
            - discount_factor: the discount factor of reward.
            - players: the number of players.
        */
        std::stack<CNode *> node_stack;
        node_stack.push(root);
        // float parent_value_prefix = 0.0;
        int is_reset = 0;
        while (node_stack.size() > 0)
        {
            CNode *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                // true_reward = node.value_prefix - (- parent_value_prefix)
                float true_reward = node->value_prefix - node->parent_value_prefix;

                if (is_reset == 1)
                {
                    true_reward = node->value_prefix;
                }
                float qsa;
                if (players == 1)
                {
                    qsa = true_reward + discount_factor * node->value();
                }
                else if (players == 2)
                {
                    // TODO(pu): why only the last reward multiply the discount_factor?
                    qsa = true_reward + discount_factor * (-1) * node->value();
                }

                min_max_stats.update(qsa);
            }

            for (auto a : node->legal_actions)
            {
                CNode *child = node->get_child(a);
                if (child->expanded())
                {
                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }

            is_reset = node->is_reset;
        }
    }

    void cbackpropagate(std::vector<CNode *> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor)
    {
        /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - to_play: which player to play the game in the current node.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
        */
        assert(to_play == -1 || to_play == 1 || to_play == 2);
        if (to_play == -1)
        {
            // for play-with-bot-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                float true_reward = node->value_prefix - parent_value_prefix;
                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset
                    true_reward = node->value_prefix;
                }

                bootstrap_value = true_reward + discount_factor * bootstrap_value;
            }
        }
        else
        {
            // for self-play-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                if (node->to_play == to_play)
                {
                    node->value_sum += bootstrap_value;
                }
                else
                {
                    node->value_sum += -bootstrap_value;
                }
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                float true_reward = node->value_prefix - parent_value_prefix;

                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset
                    true_reward = node->value_prefix;
                }
                if (node->to_play == to_play)
                {
                    bootstrap_value = -true_reward + discount_factor * bootstrap_value;
                }
                else
                {
                    bootstrap_value = true_reward + discount_factor * bootstrap_value;
                }
            }
        }
    }

    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - current_latent_state_index: The index of latent state of the leaf node in the search path.
            - discount_factor: the discount factor of reward.
            - value_prefixs: the value prefixs of nodes along the search path.
            - values: the values to propagate along the search path.
            - policies: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - is_reset_list: the vector of is_reset nodes along the search path, where is_reset represents for whether the parent value prefix needs to be reset.
            - to_play_batch: the batch of which player is playing on this node.
        */
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, i, value_prefixs[i], policies[i]);
            // reset
            results.nodes[i]->is_reset = is_reset_list[i];

            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount_factor);
        }
    }

    std::vector<int> cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players)
    {
        /*
        Overview:
            Select the child node of the roots according to ucb scores.
        Arguments:
            - root: the roots to select the child node.
            - min_max_stats: a tool used to min-max normalize the score.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - mean_q: the mean q value of the parent node.
            - players: the number of players.
        Returns:
            - action: the action vector to select.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;

        for (auto a : root->legal_actions)
        {
            CNode *child = root->get_child(a);
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount_factor, players);

            if (max_score < temp_score)
            {
                max_score = temp_score;
                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(a);
            }
        }

        if (max_index_lst.size() > 0)
        {
            int rand_index = rand() % max_index_lst.size();
            std::vector<int> result(NUM_ACTION_HEADS, -1);
            result[0] = max_index_lst[rand_index];  // Only set the first action
            return result;
        }
        else
        {
            // Return a default action vector with -1s if no valid actions found
            return std::vector<int>(NUM_ACTION_HEADS, -1);
        }
    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - is_reset: whether the value prefix needs to be reset.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - parent_value_prefix: the value prefix of parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
        Returns:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->value_prefix - parent_value_prefix;
            if (is_reset == 1)
            {
                true_reward = child->value_prefix;
            }

            if (players == 1)
            {
                value_score = true_reward + discount_factor * child->value();
            }
            else if (players == 2)
            {
                value_score = true_reward + discount_factor * (-child->value());
            }
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
        {
            value_score = 0;
        }
        else if (value_score > 1)
        {
            value_score = 1;
        }

        return prior_score + value_score; // ucb_value
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch)
    {
        /*
        Overview:
            Search node path from the roots.
        Arguments:
            - roots: the roots that search from.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - min_max_stats: a tool used to min-max normalize the score.
            - results: the search results.
            - virtual_to_play_batch: the batch of which player is playing on this node.
        */
        // set seed
        get_time_and_set_rand_seed();

        float parent_q = 0.0;

        // Initialize vectors with the correct size
        results.latent_state_index_in_search_path.clear();
        results.latent_state_index_in_batch.clear();
        results.last_actions.clear();
        results.search_lens.clear();
        results.virtual_to_play_batchs.clear();
        results.nodes.clear();

        // Initialize vectors with the correct size
        results.latent_state_index_in_search_path.resize(results.num);
        results.latent_state_index_in_batch.resize(results.num);
        results.last_actions.resize(results.num);
        results.search_lens.resize(results.num);
        results.virtual_to_play_batchs.resize(results.num);
        results.nodes.resize(results.num);
        results.search_paths.resize(results.num);

        // Initialize virtual_to_play_batchs and search_paths
        for (int i = 0; i < results.num; ++i) {
            results.virtual_to_play_batchs[i] = virtual_to_play_batch[i];
            results.search_paths[i] = std::vector<CNode*>();  // Initialize each search path vector
            results.last_actions[i] = std::vector<int>();  // Initialize each last_actions vector
        }

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end()); // 0 or 2
        if (largest_element == -1)
            players = 1;
        else
            players = 2;

        for (int i = 0; i < results.num; ++i)
        {
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                is_root = 0;
                parent_q = mean_q;

                std::vector<int> actions = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players);
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                    {
                        virtual_to_play_batch[i] = 2;
                    }
                    else
                    {
                        virtual_to_play_batch[i] = 1;
                    }
                }

                node->best_action = actions;
                // next
                node = node->get_child(actions);
                // Initialize last_actions[i] with the correct size if empty
                if (results.last_actions[i].empty()) {
                    results.last_actions[i].resize(NUM_ACTION_HEADS, -1);
                }
                // Copy each action value
                for (size_t j = 0; j < actions.size() && j < NUM_ACTION_HEADS; ++j) {
                    results.last_actions[i][j] = actions[j];
                }
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.latent_state_index_in_search_path[i] = parent->current_latent_state_index;
            results.latent_state_index_in_batch[i] = parent->batch_index;

            results.search_lens[i] = search_len;
            results.nodes[i] = node;
            results.virtual_to_play_batchs[i] = virtual_to_play_batch[i];
        }
    }
}