#include <yieldfarming.hpp>
#include "poolOwnersActions.cpp"
#include "userAction.cpp"
#include "adminActions.cpp"

yieldfarming::custompools_t yieldfarming::get_pools()
{
  return custompools_t(YIELDFARMING_ACCOUNT, YIELDFARMING_ACCOUNT.value);
}


yieldfarming::userinfo_t yieldfarming::get_userinfo(const name& user)
{
  return userinfo_t(YIELDFARMING_ACCOUNT,user.value);
}

yieldfarming::config_t::const_iterator yieldfarming::get_config()
{
  config_t config_table = config_t(YIELDFARMING_ACCOUNT,YIELDFARMING_ACCOUNT.value);
  check(config_table.begin() != config_table.end(), "Config table was not initialized");
  return config_table.begin();
}


void yieldfarming::receive_token_transfer
(
  const name&         from,
  const name&         to,
  const asset&        quantity,
  const std::string&  memo
)
{
  if (to != YIELDFARMING_ACCOUNT) 
  {
    return;
  }

  // get a contract where the tokens came from
  const name&   token_contract = get_first_receiver();
  auto custompools_table = get_pools();

  if(memo.find("set reward token:") != std::string::npos)
  {
    const uint64_t& parsed_pool_id = std::stoull(memo.substr(17));
    set_reward_token(parsed_pool_id, custompools_table, from, token_contract, quantity);
  }
  else if(memo.find("deposit pool:") != std::string::npos)
  {
    const uint64_t& parsed_pool_id = std::stoull(memo.substr(13));
    deposit_pool(custompools_table, from, parsed_pool_id, token_contract, quantity);
  }
  else if(memo.find("set staking token:") != std::string::npos)
  {
    const uint64_t parsed_pool_id = std::stoull(memo.substr(18));
    set_staking_token(custompools_table, from, parsed_pool_id, token_contract, quantity);
  }
  else if(memo.find("stake:") != std::string::npos)
  {
    const uint64_t& parsed_pool_id = std::stoull(memo.substr(6));
    stake_tokens(custompools_table, parsed_pool_id, from, token_contract, quantity);
  }
  else if(memo != "dev")
    check(false, "Invalid memo");
}

void yieldfarming::transfer_tokens(const name& reciever, const name& token_contract, const asset& quantity)
{
  action
  (
    permission_level{YIELDFARMING_ACCOUNT, name("active")},
    token_contract,
    name("transfer"),
    std::make_tuple
    (
      YIELDFARMING_ACCOUNT,
      reciever,
      quantity,
      std::string("")
    )
  ).send();
}

const asset yieldfarming::calculateYield(
  custompools_t::const_iterator custompools_table_itr,
  userinfo_t::const_iterator userinfo_table_itr,
  const uint64_t& time_now,
  const asset& quantity)
{
  const uint64_t user_time = userinfo_table_itr->time;
  const uint64_t& user_staked_time = time_now - user_time;
  const uint64_t& pool_duration = custompools_table_itr->end_time - custompools_table_itr->start_time;
  const float& reward_per_one_second = custompools_table_itr->reward.amount / pool_duration;
  const float& user_amount           = quantity.amount;
  const float& total_amount          = custompools_table_itr->total_staked_tokens.amount;
  const float& total_reward_amount   = custompools_table_itr->reward.amount;

  const float& reward = (user_amount / total_amount) * (float)user_staked_time * reward_per_one_second;
  
  return asset(reward, custompools_table_itr->reward.symbol);
}

const asset yieldfarming::auto_harvest(
    const name& user,
    custompools_t::const_iterator custompools_table_itr,
    userinfo_t::const_iterator userinfo_table_itr,
    uint64_t& time_now,
    const asset& quantity)
{
  if(time_now > custompools_table_itr->end_time)
    time_now = custompools_table_itr->end_time;

  const asset& user_reward = calculateYield(custompools_table_itr, userinfo_table_itr, time_now, quantity);
  
  if(user_reward.amount > 0)
  {
    check(user_reward < custompools_table_itr->reward,
     "User reward overdrawn total pool reward. Pleas, contact to developer team"); 
    transfer_tokens(user, custompools_table_itr->reward_tokens_contract, user_reward);
  }
  return user_reward;
}

// checks ----
void yieldfarming::owner_available_check(
  custompools_t::const_iterator custompools_table_itr,
  const name& owner,
  const std::string& error
)
{
  check(custompools_table_itr->available == false, error);
  check(custompools_table_itr->pool_owner == owner, "You do not own this pool");
}

void yieldfarming::push_participant(const uint64_t& pool_id, const name& participant)
{
  participants_t participants_table(YIELDFARMING_ACCOUNT,YIELDFARMING_ACCOUNT.value);
  auto participants_table_itr = participants_table.find(pool_id);
  if(participants_table_itr == std::end(participants_table))
  {
    participants_table.emplace(YIELDFARMING_ACCOUNT, [&](auto &new_row)
    {
      new_row.pool_id = pool_id;
      new_row.participants = {participant};
    });
  }
  else
  {
    auto find_participant = std::find(std::begin(participants_table_itr->participants), std::end(participants_table_itr->participants), participant);
    if(find_participant == std::end(participants_table_itr->participants))
    {
      participants_table.modify(participants_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
      {
        new_row.participants.push_back(participant);
      });
    }
  }
}

void yieldfarming::delete_from_participants (const uint64_t& pool_id, const std::vector<name>& participants)
{
  participants_t participants_table(YIELDFARMING_ACCOUNT,YIELDFARMING_ACCOUNT.value);
  auto participants_table_itr = participants_table.find(pool_id);

  if(participants_table_itr != std::end(participants_table))
  {
    for(const name& participant : participants)
    {
      auto find_participant = std::find(std::begin(participants_table_itr->participants), std::end(participants_table_itr->participants), participant);
      if(find_participant != std::end(participants_table_itr->participants))
      {
        participants_table.modify(participants_table_itr, YIELDFARMING_ACCOUNT, [&](auto &new_row)
        {
          new_row.participants.erase(find_participant);
        });
      }
    }
  }

  if(participants_table_itr->participants.size() == 0)
    participants_table.erase(participants_table_itr);



}



