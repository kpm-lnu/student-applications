void yieldfarming::createpool(const name& pool_owner)
{
  require_auth(pool_owner);

  auto config_itr = get_config();
  check(config_itr->pause == false, "Creating function is on pause. You can not create pool now.");

  auto custompools_table = get_pools();
  uint64_t pool_id = custompools_table.available_primary_key();
  if(pool_id == 0)
    ++pool_id;

  custompools_table.emplace(pool_owner, [&](auto &new_row)
  {
    new_row.pool_id     = pool_id;
    new_row.pool_owner  = pool_owner;
  });
}


void yieldfarming::set_reward_token(
  const uint64_t& parsed_pool_id,
  custompools_t& custompools_table,
  const name& owner,
  const name& contract,
  const asset& quantity
)
{
  auto custompools_table_itr = custompools_table.require_find(parsed_pool_id, "Could not find pool id");
  owner_available_check(custompools_table_itr, owner, std::string("This pool is already available"));

  transfer_tokens(owner, contract, quantity);

  if(custompools_table_itr->reward.amount > 0)
  {
    // transfer back pervious reward token
    transfer_tokens(owner, custompools_table_itr->reward_tokens_contract, custompools_table_itr->reward);
  }

  custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
  {
      new_row.reward = asset(0, quantity.symbol);
      new_row.reward_tokens_contract = contract;
  });
}

void yieldfarming::deposit_pool(
  custompools_t& custompools_table,
  const name& owner,
  const uint64_t& pool_id,
  const name& contract,
  const asset& quantity
)
{
  auto  config_table_itr      = get_config();
  auto  custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
  check(custompools_table_itr->reward_tokens_contract == contract, "Token contracts mismatch");
  check(custompools_table_itr->reward.symbol == quantity.symbol, "Tokens symbol mismatch");

  asset fees = asset(0, quantity.symbol);

  if(custompools_table_itr->available == true)
  {
    check(custompools_table_itr->end_time > current_time_point().sec_since_epoch(), "This pool has already finished");
    fees.amount = quantity.amount * config_table_itr->fee / 100;
    //take fee of token
  }
  check(custompools_table_itr->pool_owner == owner,
    "Only " + custompools_table_itr->pool_owner.to_string() + " can deposit this pool");

  custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
  {
    new_row.reward += (quantity - fees);
  });
}

void yieldfarming::set_staking_token(
  custompools_t& custompools_table,
  const name& owner,
  const uint64_t& pool_id,
  const name& contract,
  const asset& quantity
)
{
  auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
  check(custompools_table_itr->pool_owner == owner,
    "Only " + custompools_table_itr->pool_owner.to_string() + " can change staking token");
  check(custompools_table_itr->available == false, "Available pool can not be changed");

  transfer_tokens(owner, contract, quantity);

  custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
  {
    new_row.total_staked_tokens     = asset(0, quantity.symbol);
    new_row.staked_tokens_contract  = contract;
    new_row.min_stake_amount        = asset(0, quantity.symbol);
  });
}


void yieldfarming::poolsetup
(
  const name& owner,
  const uint64_t& pool_id,
  const uint64_t& start_time,
  const uint64_t& end_time
)
{
  require_auth(owner);

  auto custompools_table = get_pools();
  auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
  owner_available_check(custompools_table_itr, owner, std::string("This pool is already available"));

  custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
  {
    new_row.start_time = start_time;
    new_row.end_time = end_time;
  });

}
    
void yieldfarming::completepool(const name& owner, const uint64_t& pool_id)
{
  require_auth(owner);

  auto  config_table_itr       = get_config();
  auto  custompools_table      = get_pools();
  auto  custompools_table_itr  = custompools_table.require_find(pool_id, "Could not find pool id");              
  owner_available_check(custompools_table_itr, owner, std::string("This pool is already available"));
  check(custompools_table_itr->end_time > custompools_table_itr->start_time,
  "End time must be greater than the start time");
  check(custompools_table_itr->start_time > current_time_point().sec_since_epoch(),
  "Start time must be greater than the current time point");
  check(custompools_table_itr->staked_tokens_contract != "none"_n, "Staking token was not initialized");
  check(custompools_table_itr->reward_tokens_contract != "none"_n, "Reward token was not initialized");

  asset fees = asset(0, custompools_table_itr->reward.symbol);

  if(custompools_table_itr->reward.amount > 0)
    fees.amount = custompools_table_itr->reward.amount * config_table_itr->fee / 100;

  custompools_table.modify(custompools_table_itr, owner, [&](auto &new_row)
  {
    new_row.available = true;
    new_row.reward -= fees;
  });
}

void yieldfarming::setminstake(const uint64_t& pool_id, const name& owner, const asset& min_stake_amount)
{
  require_auth(owner);
  auto  custompools_table      = get_pools();
  auto  custompools_table_itr  = custompools_table.require_find(pool_id, "Could not find pool id");              
  check(custompools_table_itr->pool_owner == owner, "You do not own this pool");
  check(custompools_table_itr->staked_tokens_contract != "none"_n, "Staking token was not initialized");
  check(custompools_table_itr->total_staked_tokens.symbol == min_stake_amount.symbol, "Symbol mismatch");

  custompools_table.modify(custompools_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
  {
    new_row.min_stake_amount = min_stake_amount;
  });
}


void yieldfarming::deletepool(const name& owner, const uint64_t& pool_id)
{
  if(!has_auth(YIELDFARMING_ACCOUNT))
    require_auth(owner);
  auto custompools_table = get_pools();
  auto custompools_table_itr = custompools_table.require_find(pool_id, "Could not find pool id");
  owner_available_check(custompools_table_itr, owner, std::string("Available pool can not be deleted"));
  custompools_table.erase(custompools_table_itr);
}








