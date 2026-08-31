import type { UserRecord } from "../lib/types";
import { Drawer } from "./Drawer";

interface UserDrawerProps {
  user: UserRecord | null;
  onClose: () => void;
}

export function UserDrawer({ user, onClose }: UserDrawerProps) {
  return (
    <Drawer
      open={Boolean(user)}
      onClose={onClose}
      title={user?.username ?? "User"}
      subtitle={user ? `${user.userId} · ${user.communityId}` : undefined}
    >
      {user ? (
        <div className="space-y-6">
          <section className="panel p-5">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="section-kicker">Community archetype</p>
                <p className="mt-2 text-base font-semibold uppercase">{user.communityArchetype}</p>
              </div>
              <div>
                <p className="section-kicker">Split</p>
                <p className="mt-2 text-base font-semibold uppercase">{user.split}</p>
              </div>
              <div>
                <p className="section-kicker">Community purity</p>
                <p className="mt-2 font-mono text-lg">{user.communityPurity.toFixed(4)}</p>
              </div>
              <div>
                <p className="section-kicker">Community density</p>
                <p className="mt-2 font-mono text-lg">{user.communityDensity.toFixed(4)}</p>
              </div>
              <div>
                <p className="section-kicker">Followers</p>
                <p className="mt-2 font-mono text-lg">{user.followersCount}</p>
              </div>
              <div>
                <p className="section-kicker">Following</p>
                <p className="mt-2 font-mono text-lg">{user.followingCount}</p>
              </div>
              <div>
                <p className="section-kicker">Community clustering</p>
                <p className="mt-2 font-mono text-lg">{user.communityClustering.toFixed(4)}</p>
              </div>
              <div>
                <p className="section-kicker">Tweets</p>
                <p className="mt-2 font-mono text-lg">{user.tweetsTotal}</p>
              </div>
            </div>
          </section>

          <section className="panel p-5">
            <p className="section-kicker">Description excerpt</p>
            <p className="mt-3 text-sm leading-7 text-charcoal/75">
              {user.descriptionExcerpt || "No profile description available."}
            </p>
          </section>

          <section className="panel p-5">
            <p className="section-kicker">Triplet summary</p>
            <p className="mt-3 text-sm leading-7 text-charcoal/75">
              {user.tripletSummary || "No triplet summary available."}
            </p>
          </section>

          <section className="panel p-5">
            <p className="section-kicker">Post type ratios</p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              {Object.entries(user.postTypeRatios).map(([key, value]) => (
                <div key={key} className="rounded-2xl border border-charcoal/10 bg-paper px-4 py-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-charcoal/50">{key}</p>
                  <p className="mt-2 text-lg font-semibold">{value.toFixed(3)}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </Drawer>
  );
}
