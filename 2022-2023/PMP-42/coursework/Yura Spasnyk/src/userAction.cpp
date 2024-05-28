void yieldfarming::stake_tokens(
  custompools_t& custompools_table,
  const uint64_t& pool_id,
  const name& user,
  const name& contract,
  const asset& quantity
)
{
    uint64_t time_now = current_time_point().sec_since_epoch();
    auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
    check(custompools_table_itr->available == true, "This pool is not available");
    check(custompools_table_itr->end_time >= time_now, "This pool has already finished");
    check(custompools_table_itr->staked_tokens_contract == contract, "Token contract mismatch");
    check(custompools_table_itr->total_staked_tokens.symbol == quantity.symbol, "Token symbol mismatch");
    check(custompools_table_itr->pool_owner != user, "Pool owner can not take part in");
    check(custompools_table_itr->min_stake_amount <= quantity, "You need to transfer more than the minimum payment");

    auto userinfo_table = get_userinfo(user);
    auto userinfo_table_itr = userinfo_table.find(pool_id);

    if(custompools_table_itr->start_time > time_now)
        time_now = custompools_table_itr->start_time;

    if(userinfo_table_itr == std::end(userinfo_table))
    {
        userinfo_table.emplace(YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
            new_row.pool_id = pool_id;
            new_row.staked_tokens = quantity;
            new_row.collected_rewards = asset(0, custompools_table_itr->reward.symbol);
            new_row.time = time_now;
        });
        push_participant(pool_id, user);
    }
    else
    {
        userinfo_table.modify(userinfo_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
            new_row.staked_tokens += quantity;
        });
    }

    custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
    {
        new_row.total_staked_tokens += quantity;
    });

}

void yieldfarming::unstake(const name& user, const uint64_t& pool_id, const asset& quantity)
{
    if(!has_auth(YIELDFARMING_ACCOUNT))
        require_auth(user);

    uint64_t time_now = current_time_point().sec_since_epoch();
    auto custompools_table = get_pools();
    auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
    auto userinfo_table = get_userinfo(user);
    auto userinfo_table_itr = userinfo_table.require_find(pool_id, "User does not have staked tokens in this pool id");
    
    check(userinfo_table_itr->staked_tokens.symbol == quantity.symbol, "Token symbol mismatch");
    check(userinfo_table_itr->staked_tokens >= quantity, "User does not have enough tokens to unstake");

    if(custompools_table_itr->end_time <= time_now && quantity.amount == custompools_table_itr->total_staked_tokens.amount)
    {
        //if(custompools_table_itr->reward.amount > 0)
          //  transfer_tokens(custompools_table_itr->pool_owner, custompools_table_itr->reward_tokens_contract, custompools_table_itr->reward);
        custompools_table.erase(custompools_table_itr);
    }
    else
    {
        custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
            new_row.total_staked_tokens -= quantity;
        });
    }

    if(userinfo_table_itr->staked_tokens == quantity)
    {
        if(userinfo_table_itr->collected_rewards.amount > 0)
        {
            transfer_tokens(user, custompools_table_itr->reward_tokens_contract, userinfo_table_itr->collected_rewards);
        }
        userinfo_table.erase(userinfo_table_itr);
        delete_from_participants(pool_id, {user});
    }
    else
    {
        userinfo_table.modify(userinfo_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
            new_row.staked_tokens -= quantity;
        });
    }

    transfer_tokens(user, custompools_table_itr->staked_tokens_contract, quantity);
}

void yieldfarming::collect(const name& user, const uint64_t& pool_id)
{
    if(!has_auth(YIELDFARMING_ACCOUNT))
        require_auth(user);


    auto custompools_table = get_pools();
    auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
    auto userinfo_table = get_userinfo(user);
    auto userinfo_table_itr = userinfo_table.require_find(pool_id, "User does not have staked tokens in this pool id");

    check(userinfo_table_itr->collected_rewards.amount > 0, "You do not have nothing to collect");

    transfer_tokens(user, custompools_table_itr->reward_tokens_contract, userinfo_table_itr->collected_rewards);

    userinfo_table.modify(userinfo_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
    {
        new_row.collected_rewards -= new_row.collected_rewards;
    });
}







