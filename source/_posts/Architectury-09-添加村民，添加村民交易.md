---
title: Architectury-09-添加村民，添加村民交易
date: 2023-10-29 17:42:51
tags:
- 我的世界
- 模组
- java
cover: https://w.wallhaven.cc/full/kx/wallhaven-kxjv7d.jpg
---

# 添加村民

![image-20231101192249502](https://s2.loli.net/2023/11/01/gFCvuDV3jacRL2w.png)

![image-20231101192306590](https://s2.loli.net/2023/11/01/84FLulXN1qeEf6H.png)

![image-20231101192318126](https://s2.loli.net/2023/11/01/5rITKJ6wH4Q7Wlo.png)

![image-20231101192327155](https://s2.loli.net/2023/11/01/arsgJvw8TetW59S.png)

# 添加村民职业和工作方块

```java

package net.tutorialmod.villager;

import com.google.common.collect.ImmutableSet;
import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.sounds.SoundEvents;
import net.minecraft.world.entity.ai.village.poi.PoiType;
import net.minecraft.world.entity.npc.VillagerProfession;
import net.minecraft.world.entity.npc.VillagerType;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.block.ModBlock;

public class ModVillagers {
    public static final ResourceKey<PoiType> JUMPY_MASTER_KEY = createKey("jumpy_block_poi");

    public static final DeferredRegister<VillagerProfession> VILLAGER_PROFESSIONS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.VILLAGER_PROFESSION);
    public static final DeferredRegister<PoiType> POI_TYPES = DeferredRegister.create(TutorialMod.MOD_ID, Registries.POINT_OF_INTEREST_TYPE);


    public static final RegistrySupplier<PoiType> JUMPY_BLOCK_POI = POI_TYPES.register("jumpy_block_poi",
            () -> new PoiType(ImmutableSet.copyOf(ModBlock.JUMPY_BLOCK.get().getStateDefinition().getPossibleStates()),
                    1, 1));

    public static final RegistrySupplier<VillagerProfession> JUMP_MASTER = VILLAGER_PROFESSIONS.register("jumpy_master",
            () -> new VillagerProfession("jumpy_master", x -> x.value() == JUMPY_BLOCK_POI.get(),
                    x -> x.value() == JUMPY_BLOCK_POI.get(), ImmutableSet.of(), ImmutableSet.of(),
                    SoundEvents.VILLAGER_WORK_ARMORER));

    private static ResourceKey<PoiType> createKey(String pName) {
        return ResourceKey.create(Registries.POINT_OF_INTEREST_TYPE, new ResourceLocation(TutorialMod.MOD_ID,pName));
    }


    public static void register() {
        POI_TYPES.register();
        VILLAGER_PROFESSIONS.register();
    }

    


}

```





```java

package net.tutorialmod;

import com.google.common.base.Suppliers;
import dev.architectury.registry.CreativeTabRegistry;
import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrarManager;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.tutorialmod.block.ModBlock;
import net.tutorialmod.item.ModCreativeTab;
import net.tutorialmod.item.ModItem;
import net.tutorialmod.painting.ModPainting;
import net.tutorialmod.villager.ModVillagers;

import java.util.function.Supplier;

public class TutorialMod {
    public static final String MOD_ID = "tutorialmod";
    // We can use this if we don't want to use DeferredRegister
    public static final Supplier<RegistrarManager> REGISTRIES = Suppliers.memoize(() -> RegistrarManager.get(MOD_ID));

    public static void init() {
        ModCreativeTab.register();
        ModBlock.register();
        ModItem.register();
        ModPainting.register();
        ModVillagers.register();

        System.out.println(TutorialModExpectPlatform.getConfigDirectory().toAbsolutePath().normalize().toString());
    }
}

```



# 添加村民交易内容

fabric

```java
package net.tutorialmod.fabric.villager;

import net.fabricmc.fabric.api.object.builder.v1.trade.TradeOfferHelper;
import net.minecraft.world.entity.npc.VillagerProfession;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.trading.MerchantOffer;
import net.tutorialmod.item.ModItem;
import net.tutorialmod.villager.ModVillagers;

public class ModFabricVillager {
    public static void registerVillagerTrades(){
        TradeOfferHelper.registerVillagerOffers(ModVillagers.JUMP_MASTER.get(),1,
                factories->{
                    factories.add((entity, randomSource) -> new MerchantOffer(
                            new ItemStack(Items.EMERALD,3),
                            new ItemStack(ModItem.EXAMPLE_ITEM.get()),
                            6,2,0.02f
                    ));
                });

        TradeOfferHelper.registerVillagerOffers(VillagerProfession.ARMORER,1,
                factories->{
                    factories.add((entity, randomSource) -> new MerchantOffer(
                            new ItemStack(Items.EMERALD,3),
                            new ItemStack(ModItem.BLUEBERRY.get()),
                            6,2,0.02f
                    ));
                });


    }
}

```

```java

package net.tutorialmod.fabric;

import net.tutorialmod.TutorialMod;
import net.fabricmc.api.ModInitializer;
import net.tutorialmod.fabric.villager.ModFabricVillager;
import net.tutorialmod.villager.ModVillagers;

public class TutorialModFabric implements ModInitializer {
    @Override
    public void onInitialize() {
        TutorialMod.init();
        ModFabricVillager.registerVillagerTrades();
    }
}

```

forge

```java
package net.tutorialmod.forge.event;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import net.minecraft.world.entity.npc.VillagerProfession;
import net.minecraft.world.entity.npc.VillagerTrades;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.trading.MerchantOffer;
import net.minecraftforge.event.village.VillagerTradesEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.item.ModItem;
import net.tutorialmod.villager.ModVillagers;

import java.util.List;

public class ModEvents {
    @Mod.EventBusSubscriber(modid = TutorialMod.MOD_ID)
    public static class ForgeEvent{
        @SubscribeEvent
        public static void addCustomTrades(VillagerTradesEvent event){
            if(event.getType() == VillagerProfession.TOOLSMITH) {
                Int2ObjectMap<List<VillagerTrades.ItemListing>> trades = event.getTrades();
                ItemStack stack = new ItemStack(ModItem.EXAMPLE_ITEM.get(), 1);
                int villagerLevel = 1;

                trades.get(villagerLevel).add((trader, rand) -> new MerchantOffer(
                        new ItemStack(Items.EMERALD, 2),
                        stack,10,8,0.02F));
            }

            if(event.getType() == ModVillagers.JUMP_MASTER.get()) {
                Int2ObjectMap<List<VillagerTrades.ItemListing>> trades = event.getTrades();
                ItemStack stack = new ItemStack(ModItem.BLUEBERRY.get(), 15);
                int villagerLevel = 1;

                trades.get(villagerLevel).add((trader, rand) -> new MerchantOffer(
                        new ItemStack(Items.EMERALD, 5),
                        stack,10,8,0.02F));
            }
        }

    }
}

```



# 添加tag

```json
{
  "values": [
    "tutorialmod:jumpy_block_poi"
  ]
}
```

# 添加贴图



# 



