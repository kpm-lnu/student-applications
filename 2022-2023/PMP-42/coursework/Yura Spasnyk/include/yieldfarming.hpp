#include <eosio/eosio.hpp>
#include <eosio/asset.hpp>
#include <eosio/singleton.hpp>
using namespace eosio;

static constexpr name YIELDFARMING_ACCOUNT = name("pooooooooool");

class [[eosio::contract]] yieldfarming : public contract 
{
  public:
    using contract::contract;

    [[eosio::on_notify("*::transfer")]] void receive_token_transfer(
      const name&         from,
      const name&         to,
      const asset&        quantity,
      const std::string&  memo
    );

  // ------------------------------------------ ADMIN ACTIONS -------------------------------------------

    [[eosio::action]]
    void setconfig(
      const uint64_t& min_claim_time,
      const float&    fee,
      const bool&     pause
    );

    [[eosio::action]]
    void deleteconf();

    [[eosio::action]]
    void clearpool(const uint64_t& pool_id); // can be called when pool is over

    [[eosio::action]]
    void updatepool(const uint64_t& pool_id);

  // ----------------------------------------------------------------------------------------------------

  // ------------------------------------------------ POOL OWNERS ---------------------------------------

    [[eosio::action]]
    void createpool(const name& pool_owner);
  
    [[eosio::action]]
    void deletepool(const name& owner, const uint64_t& pool_id);

    [[eosio::action]]
    void poolsetup(
      const name& owner,
      const uint64_t& pool_id,
      const uint64_t& start_time,
      const uint64_t& end_time
    );

    [[eosio::action]]
    void setminstake(const uint64_t& pool_id, const name& owner, const asset& min_stake_amount);
    
    [[eosio::action]]
    void completepool(const name& owner, const uint64_t& pool_id);

  // ----------------------------------------------------------------------------------------------------

  // ---------------------------------------------- USERS -----------------------------------------------

    [[eosio::action]]
    void unstake(const name& user, const uint64_t& pool_id, const asset& quantity);

    [[eosio::action]]
    void collect(const name& user, const uint64_t& pool_id);
  // ----------------------------------------------------------------------------------------------------

  private:

    //scope: pool owner
    struct [[eosio::table]] custompools_j    // 193 bytes (first scope 305 bytes)
    {
      uint64_t  pool_id;
      name      pool_owner;
      name      reward_tokens_contract   = "none"_n;  // setting when you have transferred funds *(can be changed while available == false)
      asset     reward                  = asset(0,symbol("NONE",16));
      asset     total_staked_tokens     = asset(0,symbol("NONE",16));
      asset     min_stake_amount        = asset(0, symbol("NONE",16));
      name      staked_tokens_contract  = "none"_n;  // pool owner have to transfer reward token to set this data
      uint64_t  start_time              = 0;
      uint64_t  end_time                = 0;
      bool      available               = false;  // once you have marked as available you will no longer be able to edit and delete the pool

      uint64_t  primary_key() const { return pool_id; }
    };
    typedef multi_index<name("custompools"), custompools_j> custompools_t;

    //scope:user
    struct [[eosio::table]] userinfo_j
    {
      uint64_t  pool_id;
      uint64_t  time;
      asset     staked_tokens;
      asset     collected_rewards;

      uint64_t primary_key()const { return pool_id; }
    };
    typedef multi_index<name("userinfo"), userinfo_j> userinfo_t;

    //scope:contract
    struct [[eosio::table]] participants_j
    {
      uint64_t          pool_id;
      std::vector<name> participants;
      uint64_t primary_key()const { return pool_id; }
    };
    typedef multi_index<name("participants"), participants_j> participants_t;
    
    void push_participant(const uint64_t& pool_id, const name& participant);
    void delete_from_participants(const uint64_t& pool_id, const std::vector<name>& participants);

    //scope:contract
    struct [[eosio::table]] config_j
    {
      uint64_t  key = 777;
      uint64_t  min_claim_time;
      float     fee;
      bool      pause;

      uint64_t primary_key()const { return key; }
    };
    typedef multi_index<name("config"), config_j> config_t;

    config_t::const_iterator   get_config();
    custompools_t              get_pools();
    userinfo_t                 get_userinfo(const name& user);

    void set_reward_token(
      const uint64_t& parsed_pool_id,
      custompools_t& custompools_table,
      const name& owner,
      const name& contract,
      const asset& quantity
    );

    void deposit_pool(
      custompools_t& custompools_table,
      const name& owner,
      const uint64_t& pool_id,
      const name& contract,
      const asset& quantity
    );

    void set_staking_token(
      custompools_t& custompools_table,
      const name& owner,
      const uint64_t& pool_id,
      const name& contract,
      const asset& quantity
    );

    void stake_tokens(
      custompools_t& custompools_table,
      const uint64_t& pool_id,
      const name& user,
      const name& contract,
      const asset& quantity
    );

    void transfer_tokens(const name& reciever, const name& token_contract, const asset& quantity);

    const asset calculateYield(
      custompools_t::const_iterator custompools_table_itr,
      userinfo_t::const_iterator userinfo_table_itr,
      const uint64_t& time_now,
      const asset& quantity);
    
    const asset auto_harvest(
      const name& user,
      custompools_t::const_iterator custompools_table_itr,
      userinfo_t::const_iterator userinfo_table_itr,
      uint64_t& time_now,
      const asset& quantity);

    // -------------------------------------------------- checks -------------------------------
    // check for owner and check for available pool
    void owner_available_check(
      custompools_t::const_iterator custompools_table_itr,
      const name& owner,
      const std::string& error);

    
};
