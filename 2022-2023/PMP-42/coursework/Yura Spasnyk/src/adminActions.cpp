
void yieldfarming::updatepool(const uint64_t& pool_id)
{
    require_auth(YIELDFARMING_ACCOUNT);

    auto custompools_table = get_pools();
    auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
    participants_t participants_table(YIELDFARMING_ACCOUNT,YIELDFARMING_ACCOUNT.value);
    auto participants_table_itr = participants_table.find(pool_id);

    const uint64_t& time_now = current_time_point().sec_since_epoch();
    const uint64_t& time_to_calculate = time_now > custompools_table_itr->end_time ? custompools_table_itr->end_time : time_now;

    if(participants_table_itr != std::end(participants_table))
    {
        asset total_rewards = asset(0, custompools_table_itr->reward.symbol);
        for(const name& participant : participants_table_itr->participants)
        {
            auto userinfo_table = get_userinfo(participant);
            auto userinfo_table_itr = userinfo_table.require_find(pool_id,
            (participant.to_string() + " does not have staked tokens in this pool id").c_str());

            if(userinfo_table_itr->time < custompools_table_itr->end_time)
            {
                const asset participant_reward = calculateYield(custompools_table_itr, userinfo_table_itr, time_to_calculate, userinfo_table_itr->staked_tokens);
                total_rewards += participant_reward;

                if(participant_reward.amount > 0)
                {
                    userinfo_table.modify(userinfo_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
                    {
                        new_row.collected_rewards += participant_reward;
                        new_row.time               = time_to_calculate;
                    });
                }
            }
        }
        check(total_rewards.amount <= custompools_table_itr->reward.amount, "The calculated rewards of the participants exceeds the reward of the pool");
    }
}

void yieldfarming::clearpool(const uint64_t& pool_id)
{
    require_auth(YIELDFARMING_ACCOUNT);

    auto custompools_table = get_pools();
    auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
    participants_t participants_table(YIELDFARMING_ACCOUNT,YIELDFARMING_ACCOUNT.value);
    auto participants_table_itr = participants_table.find(pool_id);
    asset check_value = asset(0, custompools_table_itr->reward.symbol);

    check(custompools_table_itr->end_time <= current_time_point().sec_since_epoch(), "This pool is not over");

    if(participants_table_itr != std::end(participants_table))
    {
        for(const name& participant : participants_table_itr->participants)
        {
            auto userinfo_table = get_userinfo(participant);
            auto userinfo_table_itr = userinfo_table.require_find(pool_id,
            (participant.to_string() + " does not have staked tokens in this pool id").c_str());
            
            const asset participant_reward = userinfo_table_itr->collected_rewards + calculateYield(custompools_table_itr, userinfo_table_itr, custompools_table_itr->end_time, userinfo_table_itr->staked_tokens) ;

            if(participant_reward.amount > 0)
                transfer_tokens(participant, custompools_table_itr->reward_tokens_contract, participant_reward);

            transfer_tokens(participant, custompools_table_itr->staked_tokens_contract, userinfo_table_itr->staked_tokens);
            check_value += participant_reward;

            userinfo_table.erase(userinfo_table_itr);
        }
        check(check_value <= custompools_table_itr->reward, "The calculated rewards of the participants exceeds the reward of the pool");
        participants_table.erase(participants_table_itr);
    }
    const asset rest_tokens = custompools_table_itr->reward - check_value;
    if(rest_tokens.amount > 0)
        transfer_tokens(custompools_table_itr->pool_owner, custompools_table_itr->reward_tokens_contract, rest_tokens);

    custompools_table.erase(custompools_table_itr);
}

void yieldfarming::setconfig(
    const uint64_t& min_claim_time,
    const float&    fee,
    const bool&     pause
)
{
    require_auth(YIELDFARMING_ACCOUNT);

    config_t config_table(YIELDFARMING_ACCOUNT, YIELDFARMING_ACCOUNT.value);
    auto config_table_itr = config_table.begin();

    if(config_table_itr == std::end(config_table))
    {
        config_table.emplace(YIELDFARMING_ACCOUNT,[&](auto &new_row)
        {
            new_row.min_claim_time  = min_claim_time;
            new_row.fee             = fee;
            new_row.pause           = pause;
        });
    }
    else
    {
        config_table.modify(config_table_itr,  YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
            new_row.min_claim_time  = min_claim_time;
            new_row.fee             = fee;
            new_row.pause           = pause;
        });
    }
}

void yieldfarming::deleteconf()
{
    require_auth(YIELDFARMING_ACCOUNT);

    config_t config_table(YIELDFARMING_ACCOUNT, YIELDFARMING_ACCOUNT.value);
    auto config_table_itr = config_table.begin();
    check(config_table_itr != std::end(config_table), "Config table is empty");
    
    config_table.erase(config_table_itr);
}




